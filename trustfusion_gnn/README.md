# TrustFusion-GNN

TrustFusion-GNN is the cloud-side ML component of the agriculture IoT pipeline. It implements trustworthy multi-sensor fusion using a three-stage Graph Neural Network that estimates per-sensor trust scores, fuses readings across sensor groups, and quantifies output uncertainty.

## Workflow

1. Place collected sensor datasets (`.npz` format) in `collected_data/`.
2. Train the model with `train.py`.
3. Run inference via `InferenceEngine` in `inference.py`.
4. Inspect training results under `collected_data/training_runs/`.

## Entry Points

### `train.py`

Trains TrustFusion-GNN from pre-exported `.npz` dataset files.

```bash
python train.py \
  --data-dir collected_data \
  --epochs 50 \
  --patience 10 \
  --batch-size 64
```

Outputs training logs and a summary JSON to `collected_data/training_runs/`.

### `main.py`

Interactive demo showing model architecture and sensor graph topology.

```bash
python main.py
```

## Core Files

### Configuration

- **`config.py`** — Sensor metadata, model hyperparameters, physical constraints, thresholds.
- **`data_structures.py`** — Typed containers for system inputs, outputs, and intermediate stages.

### Data Loading

- **`dataset_loader.py`** — Loads `.npz` splits into PyTorch datasets and dataloaders.
- **`normalization.py`** — Min-max normalization and denormalization for inputs, outputs, and uncertainties.

### Graph

- **`graph_builder.py`** — Builds sensor adjacency matrices (distance-based, type-based, ESP32-group, combined, k-NN).

### Model (three-stage GNN)

- **`models/stage1_feature.py`** — Temporal encoding, statistical feature extraction, initial trust estimation.
- **`models/stage2_graph.py`** — Trust-aware graph attention, multi-hop message passing, learned adjacency refinement.
- **`models/stage3_fusion.py`** — Trust-weighted fusion, uncertainty estimation, anomaly detection, system-confidence output.
- **`models/trustfusion_gnn.py`** — Wraps all three stages into the full model.

### Training and Inference

- **`trainer.py`** — Training loop, early stopping, validation, logging.
- **`losses.py`** — Composite loss: fusion + trust + anomaly + consistency + uncertainty calibration.
- **`metrics.py`** — MAE, RMSE, MAPE, per-channel errors, anomaly AUC/F1, trust-score MAE, system confidence.
- **`inference.py`** — Single-step and full-window inference; converts raw model outputs to application-layer results.

## Collected Data

`collected_data/` holds the sensor dataset splits and training artifacts.

```
collected_data/
├── train.npz              # Training split
├── val.npz                # Validation split
├── test.npz               # Test split
├── train_summary.json     # Per-sensor statistics for training split
├── val_summary.json
├── test_summary.json
├── train_faults.json      # Per-sample anomaly labels for training split
├── val_faults.json
├── test_faults.json
├── manifest.json          # Dataset configuration and split metadata
├── analysis/              # Quality report and visualisation charts
└── training_runs/         # Per-run logs, loss histories, and metric summaries
```

Each `.npz` file contains:

| Array | Shape | Description |
|---|---|---|
| `X` | (N_samples, 7, 60, 1) | Sensor readings — 7 sensors, 60-point windows |
| `fusion_target` | (N_samples, 60, 4) | Ground-truth fused values (temp, humidity, soil, light) |
| `fault_mask` | (N_samples, 7, 60) | Per-sensor anomaly labels (1 = anomalous) |
| `credibility_target` | (N_samples, 7) | Per-sensor trust targets |
| `sensor_ids` | (7,) | Sensor identifier strings |