# TrustFusion-GNN Agriculture IoT Pipeline

## 1. Architecture Overview

The project uses a three-layer architecture:

1. Sensor Layer (ESP32 nodes)
2. Edge Layer (Raspberry Pi gateway)
3. Cloud Layer (TrustFusion-GNN model service)

### 1.1 Sensor Layer (ESP32)

Each ESP32 node samples:

- Temperature (DHT22)
- Humidity (DHT22)
- Soil moisture (capacitive sensor)
- Light intensity (BH1750)

The node sends data via MQTT in JSON format.

### 1.2 Edge Layer (Raspberry Pi)

The gateway performs:

- MQTT data ingestion
- Basic validation and quality scoring
- Edge anomaly checks
- Local persistence (SQLite)
- Batch cloud upload

### 1.3 Cloud Layer

Cloud services perform:

- Multi-sensor trust-aware fusion
- Sensor trust estimation
- Uncertainty estimation
- Alert generation and dashboard outputs

## 2. End-to-End Data Flow

1. ESP32 publishes sensor packets to MQTT topics.
2. Raspberry Pi subscribes and receives packets.
3. Data processor validates values and computes stats.
4. Edge detector marks suspicious readings.
5. Local storage persists records for reliability.
6. Cloud uploader pushes batches to API endpoints.
7. TrustFusion-GNN service returns fused outputs and trust scores.
8. Monitoring/dashboard layers display results and alerts.

## 3. MQTT Payload Example

```json
{
  "node_id": "node_01",
  "timestamp": 1710000000,
  "quality": 0.98,
  "sensors": {
    "temperature": {"value": 25.4, "unit": "celsius", "valid": true},
    "humidity": {"value": 62.1, "unit": "percent", "valid": true},
    "soil_moisture": {"value": 43.2, "unit": "percent", "valid": true},
    "light": {"value": 8200, "unit": "lux", "valid": true}
  }
}
```

## 4. Deployment Steps

### 4.1 Raspberry Pi Gateway

```bash
cd ~/iot_gateway\(pi\)
pip3 install -r requirements.txt
mkdir -p config logs data
python3 main.py
```

### 4.2 Optional Service Mode

Use `systemd` to run the gateway as a persistent service.

## 5. Reliability Considerations

1. Keep local buffering enabled for offline periods.
2. Enable retry logic for failed cloud uploads.
3. Monitor queue size and upload latency.
4. Rotate logs and cap database size.
5. Use health checks for MQTT broker and API endpoints.

## 6. Security Recommendations

1. Protect MQTT with authentication/TLS.
2. Store API keys securely (not hardcoded in source).
3. Restrict network access between edge and cloud.
4. Add request signing or token validation on cloud APIs.

## 7. Model Integration Notes

TrustFusion-GNN should consume time-windowed multi-sensor inputs and output:

- Fused target values
- Per-sensor trust scores
- Per-output uncertainty
- Anomaly indicators

These outputs can drive irrigation control, fault maintenance, and alert workflows.
