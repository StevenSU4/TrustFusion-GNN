# src/anomaly_detector.py
"""
Anomaly detection module
Performs preliminary anomaly detection at the edge
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Edge-side anomaly detector
    
    Uses multiple methods to detect sensor anomalies:
    1. Range check - value outside valid range
    2. Statistical check - Z-score anomaly
    3. Spike check - abrupt short-term change
    4. Consistency check - compare with same-type sensors
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize anomaly detector
        
        Args:
            config: configuration dictionary
        """
        self.config = config or {}
        
        # Default valid ranges
        self.valid_ranges = {
            "temperature": (-40, 80),
            "humidity": (0, 100),
            "soil_moisture": (0, 100),
            "light": (0, 100000)
        }
        
        # Historical data buffer
        self.history: Dict[str, deque] = {}
        self.window_size = self.config.get("window_size", 20)
        
        # Z-score threshold
        self.z_threshold = self.config.get("z_threshold", 3.0)
        
        # Spike threshold (multiple of standard deviation)
        self.spike_threshold = self.config.get("spike_threshold", 2.0)
        
        # Anomaly counter (for persistent anomaly detection)
        self.anomaly_counts: Dict[str, int] = {}
    
    def detect(
        self, 
        node_id: str, 
        sensor_type: str, 
        value: float
    ) -> Tuple[bool, List[str]]:
        """
        Detect whether a single data point is anomalous
        
        Args:
            node_id: node ID
            sensor_type: sensor type
            value: sensor value
            
        Returns:
            (is_anomaly, anomaly_reason_list)
        """
        anomalies = []
        key = f"{node_id}_{sensor_type}"
        
        # 1. Range check
        if sensor_type in self.valid_ranges:
            min_val, max_val = self.valid_ranges[sensor_type]
            if value < min_val or value > max_val:
                anomalies.append(f"out_of_range: {value} not in [{min_val}, {max_val}]")
        
        # Get historical data
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
        
        history = list(self.history[key])
        
        # 2. Statistical anomaly detection (requires enough history)
        if len(history) >= 5:
            mean = np.mean(history)
            std = np.std(history)
            
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > self.z_threshold:
                    anomalies.append(f"statistical: z_score={z_score:.2f}")
        
        # 3. Spike detection
        if len(history) >= 1:
            last_value = history[-1]
            change = abs(value - last_value)
            
            # Compute allowed maximum change
            if len(history) >= 3:
                typical_change = np.std(np.diff(history)) if len(history) > 1 else 0
                if typical_change > 0 and change > self.spike_threshold * typical_change:
                    anomalies.append(f"spike: change={change:.2f}")
            else:
                # Use default threshold
                default_thresholds = {
                    "temperature": 5.0,
                    "humidity": 20.0,
                    "soil_moisture": 30.0,
                    "light": 10000.0
                }
                threshold = default_thresholds.get(sensor_type, float("inf"))
                if change > threshold:
                    anomalies.append(f"spike: change={change:.2f}")
        
        # Update historical data
        self.history[key].append(value)
        
        # Update anomaly counter
        if anomalies:
            self.anomaly_counts[key] = self.anomaly_counts.get(key, 0) + 1
        else:
            self.anomaly_counts[key] = 0
        
        return len(anomalies) > 0, anomalies
    
    def detect_cross_sensor(
        self, 
        sensor_type: str, 
        values: Dict[str, float]
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Cross-sensor consistency detection
        
        Args:
            sensor_type: sensor type
            values: {node_id: value} dictionary
            
        Returns:
            {node_id: (is_anomaly, anomaly_reasons)} dictionary
        """
        if len(values) < 2:
            return {}
        
        results = {}
        all_values = list(values.values())
        median = np.median(all_values)
        mad = np.median([abs(v - median) for v in all_values])  # Median Absolute Deviation
        
        if mad == 0:
            mad = np.std(all_values) / 1.4826  # Fallback estimate from std
        
        if mad == 0:
            return {}
        
        for node_id, value in values.items():
            # Compute modified Z-score
            modified_z = 0.6745 * (value - median) / mad
            
            if abs(modified_z) > self.z_threshold:
                results[node_id] = (
                    True, 
                    [f"inconsistent: modified_z={modified_z:.2f}, median={median:.2f}"]
                )
            else:
                results[node_id] = (False, [])
        
        return results
    
    def is_sensor_faulty(self, node_id: str, sensor_type: str) -> bool:
        """
        Determine whether a sensor is likely faulty
        
        Consecutive anomalies indicate possible fault
        """
        key = f"{node_id}_{sensor_type}"
        return self.anomaly_counts.get(key, 0) >= 5
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all sensors"""
        status = {}
        
        for key, count in self.anomaly_counts.items():
            node_id, sensor_type = key.rsplit("_", 1)
            
            if node_id not in status:
                status[node_id] = {}
            
            status[node_id][sensor_type] = {
                "consecutive_anomalies": count,
                "possibly_faulty": count >= 5,
                "history_size": len(self.history.get(key, []))
            }
        
        return status