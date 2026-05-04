# src/mqtt_handler.py
"""
MQTT handling module
Responsible for MQTT communication with ESP32 sensor nodes
"""

import json
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTHandler:
    """
    MQTT client handler
    
    Features:
    1. Connect to MQTT broker
    2. Subscribe to sensor-data topics
    3. Parse received JSON payloads
    4. Invoke callback for downstream handling
    """
    
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        username: str = "",
        password: str = ""
    ):
        """
        Initialize MQTT handler
        
        Args:
            broker: MQTT broker address
            port: MQTT port
            username: username (optional)
            password: password (optional)
        """
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        
        # Create MQTT client
        self.client = mqtt.Client(client_id="raspberry_gateway")
        
        # Register callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Set auth (if provided)
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Data callback (set externally)
        self.data_callback: Optional[Callable] = None
        
        # Subscribed topic list
        self.topics: list = []
        
        # Connection status
        self.connected = False
        
        # Metrics
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0,
            "last_message_time": None
        }
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()  # start background network loop
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect MQTT"""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str):
        """Subscribe a topic"""
        self.topics.append(topic)
        if self.connected:
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message"""
        try:
            payload = json.dumps(message)
            self.client.publish(topic, payload)
            logger.debug(f"Published to {topic}: {payload[:100]}...")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
    
    def set_data_callback(self, callback: Callable):
        """Set data-processing callback"""
        self.data_callback = callback
    
    def _on_connect(self, client, userdata, flags, rc):
        """Connect callback"""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker successfully")
            
            # Re-subscribe all topics
            for topic in self.topics:
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect, return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Disconnect callback"""
        self.connected = False
        logger.warning(f"Disconnected from MQTT broker, rc: {rc}")
        
        # Try reconnecting
        if rc != 0:
            logger.info("Attempting to reconnect...")
    
    def _on_message(self, client, userdata, msg):
        """Message callback"""
        try:
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = datetime.now()
            
            # Parse JSON
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            logger.debug(f"Received message on {topic}")
            
            # Add gateway receive timestamp
            data["gateway_timestamp"] = datetime.now().isoformat()
            data["topic"] = topic
            
            # Call processing callback
            if self.data_callback:
                self.data_callback(data)
                self.stats["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.stats.copy()