# TrustFusion-GNN: Trustworthy Multi-Sensor Fusion for Smart Agriculture IoT

> Course project for **IERG5230** @ CUHK (2026)
> 
> by Yihang SU (1155173753) · Bohan LIU (1155173765)

A three-layer IoT system that collects real-time agricultural sensor data from ESP32 edge nodes, preprocesses it on a Raspberry Pi gateway, and runs a trust-aware Graph Neural Network (GNN) in the cloud to fuse multi-sensor readings with per-sensor trust estimation and uncertainty quantification.

## System Architecture

```
[ESP32 Sensor Nodes]  →  [Raspberry Pi Gateway]  →  [TrustFusion-GNN Cloud Model]
   Sensor Layer               Edge Layer                    Cloud Layer
```

### Sensor Layer (ESP32 Nodes)

Three ESP32 boards host seven sensors:

| Sensor ID | Type | Model | ESP32 |
|---|---|---|---|
| `temp_1` | Temperature | DHT22 | #1 |
| `humidity_1` | Humidity | DHT22 | #1 |
| `temp_2` | Temperature | DHT22 | #2 |
| `humidity_2` | Humidity | DHT22 | #2 |
| `soil_1` | Soil Moisture | Capacitive v1.2 | #3 |
| `soil_2` | Soil Moisture | Capacitive v1.2 | #3 |
| `light` | Light Intensity | BH1750 | #3 |

Each node publishes JSON packets to an MQTT broker at 1 Hz.

### Edge Layer (Raspberry Pi Gateway)

The gateway (`iot_gateway(pi)/`) performs:

- MQTT subscription and JSON parsing
- Data validation and quality scoring
- Edge-side anomaly detection (20-point sliding window)
- Local SQLite persistence
- Batch HTTP upload to cloud API every 30 seconds

### Cloud Layer (TrustFusion-GNN)

The cloud model (`trustfusion_gnn/`) runs a three-stage GNN on 60-point sensor windows:

| Stage | Function |
|---|---|
| Stage 1 | Temporal feature extraction + initial trust estimation |
| Stage 2 | Trust-aware graph attention + multi-hop message passing |
| Stage 3 | Trust-weighted fusion + uncertainty estimation + anomaly detection |

**Outputs per window:**
- Fused values: temperature (°C), humidity (%RH), soil moisture (%), light (lux)
- Per-sensor trust scores τ ∈ [0, 1]
- Per-output uncertainty σ
- Per-sensor anomaly flags and scores
- Overall system confidence

## Repository Structure

```
project/
├── iot_gateway(pi)/           # Raspberry Pi edge gateway
│   ├── main.py                # Gateway entry point
│   ├── config/config.yaml     # MQTT, cloud API, processing settings
│   ├── src/
│   │   ├── mqtt_handler.py    # MQTT client
│   │   ├── data_processor.py  # Validation and quality scoring
│   │   ├── anomaly_detector.py# Edge anomaly detection
│   │   ├── local_storage.py   # SQLite persistence
│   │   └── cloud_uploader.py  # Batch HTTP upload worker
│   └── requirements.txt
│
├── trustfusion_gnn/           # Cloud-side GNN model
│   ├── train.py               # Train from collected dataset files
│   ├── main.py                # Architecture and graph demo
│   ├── config.py              # System configuration
│   ├── dataset_loader.py      # Dataset loading utilities
│   ├── graph_builder.py       # Sensor graph construction
│   ├── trainer.py             # Training loop
│   ├── inference.py           # Runtime inference engine
│   ├── losses.py              # Composite loss functions
│   ├── metrics.py             # Evaluation metrics
│   ├── normalization.py       # Min-max normalization
│   ├── data_structures.py     # Typed data containers
│   ├── models/
│   │   ├── trustfusion_gnn.py # Full model wrapper
│   │   ├── stage1_feature.py  # Stage 1: feature extraction
│   │   ├── stage2_graph.py    # Stage 2: GNN attention
│   │   └── stage3_fusion.py   # Stage 3: fusion + uncertainty
│   ├── collected_data/        # Sensor dataset splits and training artifacts
│   └── requirements.txt
│
└── pipeline.md                # Full architecture and deployment reference
```

## Quick Start

### Raspberry Pi Gateway

```bash
cd iot_gateway(pi)
pip3 install -r requirements.txt
mkdir -p config logs data
python3 main.py
```

Edit `config/config.yaml` to point to your MQTT broker and cloud API endpoint before running.

### TrustFusion-GNN Training

```bash
cd trustfusion_gnn
pip install -r requirements.txt

# Train from collected sensor data
python train.py --data-dir collected_data --epochs 50 --patience 10 --batch-size 64
```

Training results (logs, loss histories, metric summaries) are written to `collected_data/training_runs/`.

### Architecture Demo

```bash
cd trustfusion_gnn
python main.py
```

Prints model parameter counts, output shapes, and the sensor graph topology.

## Dataset

The collected dataset is stored in `trustfusion_gnn/collected_data/` as NumPy compressed archives.

| Split | Samples | File |
|---|---|---|
| Train | 2000 | `train.npz` |
| Validation | 300 | `val.npz` |
| Test | 300 | `test.npz` |

**Window configuration:** 7 sensors × 60 timesteps × 1 feature, sampled at 1 Hz.

**Anomaly statistics:** ~8.5% of sensor readings show at least one anomaly type (bias, drift, noise, spike, stuck-at, or random outlier). ~80% of windows contain at least one anomalous sensor reading.

## Dependencies

### Raspberry Pi Gateway

```
paho-mqtt >= 1.6.0
pyyaml >= 6.0
requests >= 2.28.0
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
```

### TrustFusion-GNN

```
torch >= 1.12.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
tqdm >= 4.62.0
```

## Results

Best validation metrics from training runs on the collected dataset:

| Metric | Value |
|---|---|
| Overall MAE | 305.75 (dominated by light channel scale) |
| Temperature MAE | 1.66 °C |
| Humidity MAE | 2.37 %RH |
| Soil Moisture MAE | 1.30 % |
| Light MAE | 1217.7 lux |
| Anomaly Detection AUC | 0.790 |
| Anomaly F1 | 0.552 |
| Mean System Confidence | 0.751 |

## Hardware

- **Sensor nodes:** ESP32 DevKit × 3
- **Sensors:** DHT22 × 2 (temp/humidity), Capacitive Soil Moisture v1.2 × 2, BH1750 × 1
- **Gateway:** Raspberry Pi (any model with network access)
- **Connectivity:** MQTT over local Wi-Fi
