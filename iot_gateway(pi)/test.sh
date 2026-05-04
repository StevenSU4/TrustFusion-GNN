# Use mosquitto_pub to simulate ESP32 data publishing

mosquitto_pub -t "farm/sensors" -m '{
  "node_id": "node_test",
  "timestamp": 1234567890,
  "quality": 0.95,
  "sensors": {
    "temperature": {"value": 25.5, "unit": "celsius", "valid": true},
    "humidity": {"value": 60.0, "unit": "percent", "valid": true},
    "soil_moisture": {"value": 45.0, "unit": "percent", "valid": true},
    "light": {"value": 10000, "unit": "lux", "valid": true}
  },
  "device_status": {
    "wifi_rssi": -50,
    "free_heap": 200000,
    "uptime": 3600
  }
}'