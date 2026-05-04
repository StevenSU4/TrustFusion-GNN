"""Utilities for loading collected sensor datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SensorDataset(Dataset):
    """PyTorch dataset backed by collected sensor data exported as `.npz` files."""

    def __init__(self, npz_path: str):
        npz = np.load(npz_path, allow_pickle=True)
        self.X = torch.from_numpy(npz["X"]).float()
        self.fusion_target = torch.from_numpy(npz["fusion_target"]).float()
        self.fault_mask = torch.from_numpy(npz["fault_mask"]).float()
        self.credibility_target = torch.from_numpy(npz["credibility_target"]).float()
        self.sensor_ids = [str(item) for item in npz["sensor_ids"].tolist()]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.fusion_target[idx],
            self.fault_mask[idx],
            self.credibility_target[idx],
        )


def create_dataloader_from_npz(
    npz_path: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a dataloader from a collected sensor dataset file."""
    dataset = SensorDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_json(path: str) -> Dict:
    """Load a JSON file into a Python dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_summary_reasonableness(summary: Dict) -> Tuple[bool, Dict[str, object]]:
    """Apply lightweight sanity checks to a dataset split summary."""
    checks = {
        "has_samples": summary["sample_count"] > 0,
        "all_anomaly_types_present": len(summary.get("fault_type_counts", {})) >= 6,
        "anomaly_density_reasonable": 0.01 <= summary["fault_density"] <= 0.20,
        "credibility_target_reasonable": 0.70 <= summary["avg_credibility_target"] <= 0.99,
    }
    return all(checks.values()), checks