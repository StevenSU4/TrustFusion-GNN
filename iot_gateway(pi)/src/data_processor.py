# src/data_processor.py
"""
Data processing module
Handles data cleaning, formatting, and preliminary analysis
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class SensorDataBuffer:
    """
    Sensor data buffer
    Maintains a time-window buffer for each sensor
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize buffer
        
        Args:
            window_size: window size (keep latest N points)
        """
        self.window_size = window_size
        self.buffers: Dict[str, Dict[str, deque]] = {}
    
    def add_data(self, node_id: str, sensor_type: str, value: float):
        """Add one data point"""
        # Initialize node buffer
        if node_id not in self.buffers:
            self.buffers[node_id] = {}
        
        # Initialize sensor buffer
        if sensor_type not in self.buffers[node_id]:
            self.buffers[node_id][sensor_type] = deque(maxlen=self.window_size)
        
        # Append data
        self.buffers[node_id][sensor_type].append(value)
    
    def get_history(self, node_id: str, sensor_type: str) -> List[float]:
        """Get historical data"""
        if node_id in self.buffers and sensor_type in self.buffers[node_id]:
            return list(self.buffers[node_id][sensor_type])
        return []
    
    def get_statistics(self, node_id: str, sensor_type: str) -> Dict[str, float]:
        """Compute statistics"""
        history = self.get_history(node_id, sensor_type)
        
        if len(history) < 2:
            return {
                "mean": history[0] if history else 0,
                "std": 0,
                "min": history[0] if history else 0,
                "max": history[0] if history else 0,
                "count": len(history)
            }
        
        arr = np.array(history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(history)
        }


class DataProcessor:
    """
    Data processor
    
    Features:
    1. Data validation and cleaning
    2. Unit conversion and normalization
    3. Data-quality scoring
    4. Formatting for cloud ingestion
    """
    
    # Valid ranges per sensor type
    VALID_RANGES = {
        "temperature": {"min": -40, "max": 80, "unit": "celsius"},
        "humidity": {"min": 0, "max": 100, "unit": "percent"},
        "soil_moisture": {"min": 0, "max": 100, "unit": "percent"},
        "light": {"min": 0, "max": 100000, "unit": "lux"}
    }
    
    def __init__(self, window_size: int = 20):
        """
        Initialize data processor
        
        Args:
            window_size: time-window size
        """
        self.buffer = SensorDataBuffer(window_size)
        self.processed_count = 0
    
    def process(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process raw sensor data
        
        Args:
            raw_data: raw data received from MQTT
            
        Returns:
            processed data dictionary, or None if invalid
        """
        try:
            # 1. Extract basic fields
            node_id = raw_data.get("node_id", "unknown")
            timestamp = raw_data.get("timestamp", 0)
            device_quality = raw_data.get("quality", 1.0)
            
            # 2. Process each sensor reading
            sensors_data = raw_data.get("sensors", {})
            processed_sensors = {}
            
            for sensor_type, sensor_info in sensors_data.items():
                processed = self._process_sensor(
                    node_id, 
                    sensor_type, 
                    sensor_info
                )
                if processed:
                    processed_sensors[sensor_type] = processed
            
            # 3. Return None if no valid data
            if not processed_sensors:
                logger.warning(f"No valid sensor data from {node_id}")
                return None
            
            # 4. Build output
            result = {
                "node_id": node_id,
                "timestamp": datetime.now().isoformat(),
                "device_timestamp": timestamp,
                "device_quality": device_quality,
                "sensors": processed_sensors,
                "metadata": {
                    "gateway_id": "raspberry_pi_gateway",
                    "processing_version": "1.0"
                }
            }
            
            # 5. Add device-status info
            if "device_status" in raw_data:
                result["device_status"] = raw_data["device_status"]
            
            self.processed_count += 1
            logger.debug(f"Processed data from {node_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _process_sensor(
        self, 
        node_id: str, 
        sensor_type: str, 
        sensor_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process one sensor reading"""
        
        # Check validity flag
        if not sensor_info.get("valid", True):
            return None
        
        value = sensor_info.get("value")
        
        # Check whether value exists
        if value is None:
            return None
        
        # Get valid range
        valid_range = self.VALID_RANGES.get(sensor_type, {})
        min_val = valid_range.get("min", float("-inf"))
        max_val = valid_range.get("max", float("inf"))
        
        # 1. Range check
        in_range = min_val <= value <= max_val
        
        # 2. Add to buffer
        if in_range:
            self.buffer.add_data(node_id, sensor_type, value)
        
        # 3. Get statistics
        stats = self.buffer.get_statistics(node_id, sensor_type)
        
        # 4. Compute quality score
        quality = self._calculate_quality(value, stats, in_range)
        
        # 5. Detect anomaly
        is_anomaly = self._detect_anomaly(value, stats)
        
        return {
            "value": value,
            "unit": valid_range.get("unit", "unknown"),
            "quality": quality,
            "in_range": in_range,
            "is_anomaly": is_anomaly,
            "statistics": stats
        }
    
    def _calculate_quality(
        self, 
        value: float, 
        stats: Dict[str, float], 
        in_range: bool
    ) -> float:
        """
        Compute data-quality score
        
        Quality score is based on:
        1. Whether value is in valid range
        2. Deviation from historical mean
        3. Data stability (standard deviation)
        """
        quality = 1.0
        
        # 1. Range check
        if not in_range:
            quality *= 0.3
        
        # 2. Deviation check
        if stats["count"] > 1 and stats["std"] > 0:
            z_score = abs(value - stats["mean"]) / stats["std"]
            
            if z_score > 3:
                quality *= 0.5
            elif z_score > 2:
                quality *= 0.7
            elif z_score > 1:
                quality *= 0.9
        
        return round(quality, 3)
    
    def _detect_anomaly(
        self, 
        value: float, 
        stats: Dict[str, float], 
        threshold: float = 3.0
    ) -> bool:
        """
        Detect anomalies
        
        Z-score rule: |value - mean| > threshold * std
        """
        if stats["count"] < 5 or stats["std"] == 0:
            return False
        
        z_score = abs(value - stats["mean"]) / stats["std"]
        return z_score > threshold
    
    def get_formatted_batch(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        Format multiple records for cloud batch upload
        
        Args:
            data_list: list of processed records
            
        Returns:
            batch payload format required by cloud API
        """
        return {
            "batch_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "gateway_id": "raspberry_pi_gateway",
            "timestamp": datetime.now().isoformat(),
            "data_count": len(data_list),
            "data": data_list
        }