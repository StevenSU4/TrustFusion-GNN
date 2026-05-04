"""
Configuration Files - TrustFusion GNN System Configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import torch


class SensorType(Enum):
    """Sensor types"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SOIL_MOISTURE = "soil_moisture"
    LIGHT = "light"


@dataclass
class SensorMeta:
    """
    Sensor metadata M ∈ ℝ^{N×D}
    Corresponds to architecture fields: position, device model, install time, etc.
    """
    sensor_id: str
    sensor_type: SensorType
    position: Tuple[float, float, float]  # (x, y, z) 3D coordinates
    esp32_id: int                          # attached ESP32 ID
    install_date: str                      # installation date
    model: str                             # device model
    
    # Physical attributes
    unit: str
    min_value: float
    max_value: float
    normal_range: Tuple[float, float]
    accuracy: float                        # nominal accuracy
    noise_std: float                       # nominal noise std
    
    # Fusion group for output channel assignment
    fusion_group: int                      # 0=temperature, 1=humidity, 2=soil, 3=light


@dataclass
class SystemConfig:
    """Global system configuration"""
    
    # ========== Data dimensions (N, T, F) ==========
    num_sensors: int = 7                   # N: number of sensors
    window_size: int = 60                  # T: window length
    input_features: int = 1                # F: input feature dimension per sensor
    output_features: int = 4               # output feature dimension (temp, humidity, soil, light)
    
    # ========== Sampling parameters ==========
    sampling_rate_hz: float = 1.0          # sampling frequency
    evaluation_interval: int = 5           # evaluation interval (seconds)
    
    # ========== Metadata dimension D ==========
    meta_dim: int = 8                      # sensor metadata embedding dimension
    
    # ========== Model architecture parameters ==========
    # Stage 1: Feature extraction
    temporal_hidden_dim: int = 64
    temporal_layers: int = 2
    statistical_features: int = 8          # number of statistical features
    
    # Stage 2: Graph neural network
    gnn_hidden_dim: int = 64
    gnn_layers: int = 2
    num_attention_heads: int = 4
    
    # Stage 3: Fusion output
    fusion_hidden_dim: int = 64
    
    # Common
    dropout: float = 0.1
    
    # ========== Graph construction parameters ==========
    spatial_k_neighbors: int = 3
    correlation_threshold: float = 0.5
    use_learnable_graph: bool = True
    
    # ========== Loss weights (λ terms) ==========
    lambda_fusion: float = 1.0             # λ_fusion: fusion loss
    lambda_consistency: float = 0.3        # λ_consistency: spatiotemporal consistency
    lambda_credibility: float = 0.5        # λ_credibility: trust regularization
    
    # ========== Anomaly detection thresholds ==========
    anomaly_threshold: float = 0.4         # anomaly score threshold
    credibility_low_threshold: float = 0.3 # low trust threshold
    credibility_high_threshold: float = 0.7
    system_alert_threshold: float = 0.5    # global system-confidence alert threshold
    
    # ========== Training parameters ==========
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    
    # ========== Device ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== Sensor configuration ==========
    sensors: Dict[str, SensorMeta] = field(default_factory=dict)
    
    # ========== Physical constraints (for consistency loss) ==========
    physical_constraints: Dict = field(default_factory=dict)


def get_agricultural_config() -> SystemConfig:
    """
    Get agricultural monitoring configuration
    Hardware: 2x temp/humidity sensors, 2x soil-moisture sensors,
    1x light sensor, 3x ESP32 boards
    """
    config = SystemConfig()
    
    # Define 7 sensors
    config.sensors = {
        # ===== ESP32 #1: Temp/Humidity Sensor 1 (DHT22) =====
        "temp_1": SensorMeta(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            position=(0.0, 0.0, 0.5),
            esp32_id=1,
            install_date="2024-01-01",
            model="DHT22",
            unit="°C",
            min_value=-40.0,
            max_value=80.0,
            normal_range=(15.0, 35.0),
            accuracy=0.5,
            noise_std=0.3,
            fusion_group=0  # fused into temperature channel
        ),
        "humidity_1": SensorMeta(
            sensor_id="humidity_1",
            sensor_type=SensorType.HUMIDITY,
            position=(0.0, 0.0, 0.5),
            esp32_id=1,
            install_date="2024-01-01",
            model="DHT22",
            unit="%RH",
            min_value=0.0,
            max_value=100.0,
            normal_range=(30.0, 80.0),
            accuracy=2.0,
            noise_std=1.5,
            fusion_group=1  # fused into humidity channel
        ),
        
        # ===== ESP32 #2: Temp/Humidity Sensor 2 (DHT22) =====
        "temp_2": SensorMeta(
            sensor_id="temp_2",
            sensor_type=SensorType.TEMPERATURE,
            position=(5.0, 0.0, 0.5),
            esp32_id=2,
            install_date="2024-01-01",
            model="DHT22",
            unit="°C",
            min_value=-40.0,
            max_value=80.0,
            normal_range=(15.0, 35.0),
            accuracy=0.5,
            noise_std=0.3,
            fusion_group=0
        ),
        "humidity_2": SensorMeta(
            sensor_id="humidity_2",
            sensor_type=SensorType.HUMIDITY,
            position=(5.0, 0.0, 0.5),
            esp32_id=2,
            install_date="2024-01-01",
            model="DHT22",
            unit="%RH",
            min_value=0.0,
            max_value=100.0,
            normal_range=(30.0, 80.0),
            accuracy=2.0,
            noise_std=1.5,
            fusion_group=1
        ),
        
        # ===== ESP32 #3: Soil Moisture x2 + Light =====
        "soil_1": SensorMeta(
            sensor_id="soil_1",
            sensor_type=SensorType.SOIL_MOISTURE,
            position=(2.0, 3.0, -0.2),  # 20 cm underground
            esp32_id=3,
            install_date="2024-01-01",
            model="Capacitive_v1.2",
            unit="%",
            min_value=0.0,
            max_value=100.0,
            normal_range=(20.0, 60.0),
            accuracy=3.0,
            noise_std=2.0,
            fusion_group=2  # fused into soil-moisture channel
        ),
        "soil_2": SensorMeta(
            sensor_id="soil_2",
            sensor_type=SensorType.SOIL_MOISTURE,
            position=(3.0, 3.0, -0.2),
            esp32_id=3,
            install_date="2024-01-01",
            model="Capacitive_v1.2",
            unit="%",
            min_value=0.0,
            max_value=100.0,
            normal_range=(20.0, 60.0),
            accuracy=3.0,
            noise_std=2.0,
            fusion_group=2
        ),
        "light": SensorMeta(
            sensor_id="light",
            sensor_type=SensorType.LIGHT,
            position=(2.5, 1.5, 1.0),
            esp32_id=3,
            install_date="2024-01-01",
            model="BH1750",
            unit="lux",
            min_value=0.0,
            max_value=65535.0,
            normal_range=(100.0, 50000.0),
            accuracy=1.0,
            noise_std=200.0,
            fusion_group=3  # fused into light channel
        ),
    }
    
    config.num_sensors = len(config.sensors)
    
    # Physical constraint relationships
    config.physical_constraints = {
        # Negative correlation between temperature and humidity
        ("temp_1", "humidity_1"): {"type": "negative_corr", "strength": 0.5},
        ("temp_2", "humidity_2"): {"type": "negative_corr", "strength": 0.5},
        # Same-type sensors should be similar
        ("temp_1", "temp_2"): {"type": "similar", "strength": 0.8, "max_diff": 3.0},
        ("humidity_1", "humidity_2"): {"type": "similar", "strength": 0.7, "max_diff": 10.0},
        ("soil_1", "soil_2"): {"type": "similar", "strength": 0.6, "max_diff": 15.0},
        # Temperature and light are positively correlated during daytime
        ("temp_1", "light"): {"type": "positive_corr", "strength": 0.4, "condition": "daytime"},
    }
    
    return config