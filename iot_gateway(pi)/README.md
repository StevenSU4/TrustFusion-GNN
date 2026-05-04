# TrustFusion-GNN IoT Gateway (Raspberry Pi)

This folder contains the Raspberry Pi edge gateway used in the TrustFusion-GNN agriculture IoT pipeline.

## Responsibilities

1. Receive MQTT sensor messages from ESP32 nodes.
2. Validate and normalize incoming data.
3. Perform edge-side anomaly checks.
4. Persist records to local SQLite storage.
5. Upload batches to the cloud API.

## Folder Structure

- `main.py`: Gateway entrypoint and module orchestration.
- `config/config.yaml`: Runtime configuration.
- `src/mqtt_handler.py`: MQTT client wrapper.
- `src/data_processor.py`: Data validation and processing.
- `src/anomaly_detector.py`: Edge anomaly detection logic.
- `src/local_storage.py`: Local SQLite storage layer.
- `src/cloud_uploader.py`: Background cloud upload worker.
- `test.sh`: MQTT publish test script.
- `data/`: Local database files.
- `logs/`: Runtime log files.

## Quick Start

```bash
cd ~/iot_gateway\(pi\)
pip3 install -r requirements.txt
mkdir -p config logs data
python3 main.py
```

## Systemd Service (Optional)

Create `/etc/systemd/system/iot-gateway.service`:

```ini
[Unit]
Description=IoT Gateway Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/iot_gateway(pi)
ExecStart=/usr/bin/python3 /home/pi/iot_gateway(pi)/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable iot-gateway
sudo systemctl start iot-gateway
sudo systemctl status iot-gateway
```

## Data Flow

ESP32 Nodes -> MQTT Broker -> Raspberry Pi Gateway -> Cloud API -> TrustFusion-GNN Inference -> Visualization/Alerts

## Notes

- Keep `config/config.yaml` aligned with your MQTT and cloud deployment.
- Use secure credentials in production.
- Consider log rotation and database backup policies for long-running deployment.
