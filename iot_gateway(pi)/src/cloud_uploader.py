# src/cloud_uploader.py
"""
Cloud upload module
Responsible for uploading data to the cloud server
"""

import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time
from threading import Thread, Event

logger = logging.getLogger(__name__)


class CloudUploader:
    """
    Cloud data uploader
    
    Features:
    1. Batch upload sensor data
    2. Auto-retry failed uploads
    3. Cache data while offline
    4. Asynchronous background upload
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        upload_interval: int = 30,
        batch_size: int = 50
    ):
        """
        Initialize uploader
        
        Args:
            api_url: cloud API endpoint
            api_key: API key
            upload_interval: upload interval in seconds
            batch_size: batch size per upload
        """
        self.api_url = api_url
        self.api_key = api_key
        self.upload_interval = upload_interval
        self.batch_size = batch_size
        
        # Pending upload queue
        self.upload_queue: List[Dict[str, Any]] = []
        
        # Metrics
        self.stats = {
            "total_uploaded": 0,
            "total_failed": 0,
            "last_upload_time": None,
            "last_error": None
        }
        
        # Background upload thread control
        self._stop_event = Event()
        self._upload_thread: Optional[Thread] = None
    
    def start(self):
        """Start background upload thread"""
        self._stop_event.clear()
        self._upload_thread = Thread(target=self._upload_loop, daemon=True)
        self._upload_thread.start()
        logger.info("Cloud uploader started")
    
    def stop(self):
        """Stop background upload thread"""
        self._stop_event.set()
        if self._upload_thread:
            self._upload_thread.join(timeout=5)
        logger.info("Cloud uploader stopped")
    
    def add_to_queue(self, data: Dict[str, Any]):
        """Add data to upload queue"""
        self.upload_queue.append(data)
        logger.debug(f"Data added to upload queue, queue size: {len(self.upload_queue)}")
    
    def _upload_loop(self):
        """Background upload loop"""
        while not self._stop_event.is_set():
            try:
                # Wait for configured interval
                if self._stop_event.wait(self.upload_interval):
                    break
                
                # Check whether there is data to upload
                if self.upload_queue:
                    self._do_upload()
                    
            except Exception as e:
                logger.error(f"Error in upload loop: {e}")
    
    def _do_upload(self):
        """Perform upload operation"""
        if not self.upload_queue:
            return
        
        # Take one batch of data
        batch = self.upload_queue[:self.batch_size]
        
        try:
            # Build request payload
            payload = {
                "gateway_id": "raspberry_pi_gateway",
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(batch),
                "data": batch
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            # Send request
            response = requests.post(
                f"{self.api_url}/sensor-data/batch",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                # Upload succeeded, remove batch from queue
                self.upload_queue = self.upload_queue[len(batch):]
                self.stats["total_uploaded"] += len(batch)
                self.stats["last_upload_time"] = datetime.now().isoformat()
                logger.info(f"Successfully uploaded {len(batch)} records")
                
                # Process response (may contain server-side analysis)
                result = response.json()
                self._handle_response(result)
                
            else:
                # Upload failed
                self.stats["total_failed"] += len(batch)
                self.stats["last_error"] = f"HTTP {response.status_code}"
                logger.error(f"Upload failed: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to cloud server, data queued for later")
            self.stats["last_error"] = "Connection error"
            
        except requests.exceptions.Timeout:
            logger.warning("Upload timed out, will retry later")
            self.stats["last_error"] = "Timeout"
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            self.stats["last_error"] = str(e)
    
    def _handle_response(self, result: Dict[str, Any]):
        """Handle cloud response"""
        # Check anomaly alerts
        if "alerts" in result:
            for alert in result["alerts"]:
                logger.warning(f"Cloud alert: {alert}")
        
        # Check control commands
        if "commands" in result:
            for cmd in result["commands"]:
                logger.info(f"Cloud command received: {cmd}")
                # TODO: execute control command
    
    def upload_now(self) -> bool:
        """Upload all queued data immediately"""
        if not self.upload_queue:
            return True
        
        original_size = len(self.upload_queue)
        
        while self.upload_queue:
            self._do_upload()
            
            # If queue size does not shrink, upload failed
            if len(self.upload_queue) >= original_size:
                return False
            
            original_size = len(self.upload_queue)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get upload metrics"""
        return {
            **self.stats,
            "queue_size": len(self.upload_queue)
        }