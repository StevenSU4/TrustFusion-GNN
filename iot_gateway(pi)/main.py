#!/usr/bin/env python3
# main.py
"""
TrustFusion-GNN Raspberry Pi gateway main program

Features:
1. Receive MQTT data from ESP32 sensor nodes
2. Data processing and preliminary anomaly detection
3. Local storage
4. Cloud upload
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any

import yaml

# Add src directory to import path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mqtt_handler import MQTTHandler
from data_processor import DataProcessor
from local_storage import LocalStorage
from cloud_uploader import CloudUploader
from anomaly_detector import AnomalyDetector


class Gateway:
    """
    IoT gateway main class
    
    Integrates all modules and coordinates data flow
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize gateway
        
        Args:
            config_path: config file path
        """
        # Load config
        self.config = self._load_config(config_path)
        
        # Configure logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing IoT Gateway...")
        
        # Initialize modules
        self._init_modules()
        
        # Runtime state
        self.running = False
        
        # Metrics
        self.stats = {
            "start_time": None,
            "messages_received": 0,
            "messages_processed": 0,
            "anomalies_detected": 0
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"Config file not found: {config_path}")
            print("Creating default config...")
            self._create_default_config(config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path: str):
        """Create default config file"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "",
                "password": "",
                "topics": {
                    "sensor_data": "farm/sensors",
                    "control": "farm/control/#",
                    "status": "farm/status/#"
                }
            },
            "cloud": {
                "enabled": False,
                "api_url": "http://your-cloud-server.com/api",
                "api_key": "your-api-key",
                "upload_interval": 30
            },
            "processing": {
                "window_size": 20,
                "anomaly_threshold": 3.0
            },
            "storage": {
                "database_path": "data/sensor_data.db",
                "max_records": 100000
            },
            "logging": {
                "level": "INFO",
                "file": "logs/gateway.log"
            }
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _setup_logging(self):
        """Set up logging"""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/gateway.log")
        
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging handlers
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
    
    def _init_modules(self):
        """Initialize functional modules"""
        mqtt_config = self.config.get("mqtt", {})
        cloud_config = self.config.get("cloud", {})
        storage_config = self.config.get("storage", {})
        processing_config = self.config.get("processing", {})
        
        # MQTT handler
        self.mqtt_handler = MQTTHandler(
            broker=mqtt_config.get("broker", "localhost"),
            port=mqtt_config.get("port", 1883),
            username=mqtt_config.get("username", ""),
            password=mqtt_config.get("password", "")
        )
        
        # Data processor
        self.data_processor = DataProcessor(
            window_size=processing_config.get("window_size", 20)
        )
        
        # Local storage
        self.local_storage = LocalStorage(
            db_path=storage_config.get("database_path", "data/sensor_data.db"),
            max_records=storage_config.get("max_records", 100000)
        )
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(processing_config)
        
        # Cloud uploader
        if cloud_config.get("enabled", False):
            self.cloud_uploader = CloudUploader(
                api_url=cloud_config.get("api_url", ""),
                api_key=cloud_config.get("api_key", ""),
                upload_interval=cloud_config.get("upload_interval", 30)
            )
        else:
            self.cloud_uploader = None
            self.logger.info("Cloud upload disabled")
        
        self.logger.info("All modules initialized")
    
    def _on_sensor_data(self, raw_data: Dict[str, Any]):
        """
        Sensor-data callback.

        Data flow:
        1. Receive raw data
        2. Process data
        3. Detect anomalies
        4. Store locally
        5. Enqueue for cloud upload
        """
        self.stats["messages_received"] += 1
        
        try:
            # 1. Process data
            processed_data = self.data_processor.process(raw_data)
            
            if processed_data is None:
                self.logger.warning("Data processing failed, skipping")
                return
            
            # 2. Detect anomalies
            node_id = processed_data.get("node_id", "unknown")
            sensors = processed_data.get("sensors", {})
            
            for sensor_type, sensor_info in sensors.items():
                value = sensor_info.get("value")
                if value is not None:
                    is_anomaly, reasons = self.anomaly_detector.detect(
                        node_id, sensor_type, value
                    )
                    
                    sensor_info["edge_anomaly"] = is_anomaly
                    sensor_info["anomaly_reasons"] = reasons
                    
                    if is_anomaly:
                        self.stats["anomalies_detected"] += 1
                        self.logger.warning(
                            f"Anomaly detected: {node_id}/{sensor_type} = {value}, "
                            f"reasons: {reasons}"
                        )
            
            # 3. Store locally
            self.local_storage.save_sensor_data(processed_data)
            
            # 4. Add to cloud-upload queue
            if self.cloud_uploader:
                self.cloud_uploader.add_to_queue(processed_data)
            
            self.stats["messages_processed"] += 1
            self.logger.debug(f"Processed data from {node_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling sensor data: {e}")
    
    def start(self):
        """Start gateway"""
        self.logger.info("Starting IoT Gateway...")
        self.running = True
        self.stats["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Register MQTT callback
        self.mqtt_handler.set_data_callback(self._on_sensor_data)
        
        # Connect MQTT
        if not self.mqtt_handler.connect():
            self.logger.error("Failed to connect to MQTT broker")
            return False
        
        # Subscribe topics
        mqtt_topics = self.config.get("mqtt", {}).get("topics", {})
        self.mqtt_handler.subscribe(mqtt_topics.get("sensor_data", "farm/sensors"))
        
        # Start cloud uploader
        if self.cloud_uploader:
            self.cloud_uploader.start()
        
        self.logger.info("IoT Gateway started successfully")
        return True
    
    def stop(self):
        """Stop gateway"""
        self.logger.info("Stopping IoT Gateway...")
        self.running = False
        
        # Disconnect MQTT
        self.mqtt_handler.disconnect()
        
        # Stop cloud uploader
        if self.cloud_uploader:
            self.cloud_uploader.stop()
        
        self.logger.info("IoT Gateway stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get gateway status"""
        return {
            "running": self.running,
            "stats": self.stats,
            "mqtt": self.mqtt_handler.get_stats(),
            "storage": self.local_storage.get_statistics(),
            "cloud": self.cloud_uploader.get_stats() if self.cloud_uploader else None,
            "sensor_health": self.anomaly_detector.get_health_status()
        }
    
    def run_forever(self):
        """Run until stop signal is received"""
        if not self.start():
            return
        
        # Set signal handlers
        def signal_handler(signum, frame):
            self.logger.info("Received stop signal")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
                
                # Periodically print status
                if int(time.time()) % 60 == 0:
                    status = self.get_status()
                    self.logger.info(
                        f"Status: received={status['stats']['messages_received']}, "
                        f"processed={status['stats']['messages_processed']}, "
                        f"anomalies={status['stats']['anomalies_detected']}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")


def main():
    """Main entry function"""
    print("=" * 60)
    print("  TrustFusion-GNN IoT Gateway")
    print("  Version: 1.0.0")
    print("=" * 60)
    
    # Create and run gateway
    gateway = Gateway()
    gateway.run_forever()


if __name__ == "__main__":
    main()