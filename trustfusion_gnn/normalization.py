"""
Data normalization utilities
Unify input/output scales to prevent illumination magnitude from dominating other channels.
"""
import torch

from config import SystemConfig


class DataNormalizer:
    """Min-max normalization based on configured sensor ranges."""

    def __init__(self, config: SystemConfig):
        self.sensor_ids = list(config.sensors.keys())

        input_min = [config.sensors[sid].min_value for sid in self.sensor_ids]
        input_max = [config.sensors[sid].max_value for sid in self.sensor_ids]

        output_min = []
        output_max = []
        for group_idx in range(config.output_features):
            group_sensors = [
                s for s in config.sensors.values() if s.fusion_group == group_idx
            ]
            if not group_sensors:
                output_min.append(0.0)
                output_max.append(1.0)
                continue
            output_min.append(min(s.min_value for s in group_sensors))
            output_max.append(max(s.max_value for s in group_sensors))

        # Input: (B, N, T, F=1)
        self.input_min = torch.tensor(input_min, dtype=torch.float32).view(1, -1, 1, 1)
        self.input_scale = (
            torch.tensor(input_max, dtype=torch.float32).view(1, -1, 1, 1) - self.input_min
        ).clamp(min=1e-6)

        # Output: (B, T, output_F)
        self.output_min = torch.tensor(output_min, dtype=torch.float32).view(1, 1, -1)
        self.output_scale = (
            torch.tensor(output_max, dtype=torch.float32).view(1, 1, -1) - self.output_min
        ).clamp(min=1e-6)

    @staticmethod
    def _to(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.to(device=ref.device, dtype=ref.dtype)

    def _output_stats_for(self, y: torch.Tensor):
        if y.dim() == 3:
            out_min = self._to(self.output_min, y)
            out_scale = self._to(self.output_scale, y)
        elif y.dim() == 2:
            out_min = self._to(self.output_min.squeeze(0), y)
            out_scale = self._to(self.output_scale.squeeze(0), y)
        else:
            raise ValueError(f"Unexpected output tensor shape: {tuple(y.shape)}")
        return out_min, out_scale

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        in_min = self._to(self.input_min, x)
        in_scale = self._to(self.input_scale, x)
        return (x - in_min) / in_scale

    def normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        out_min, out_scale = self._output_stats_for(y)
        return (y - out_min) / out_scale

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        out_min, out_scale = self._output_stats_for(y)
        return y * out_scale + out_min

    def denormalize_uncertainty(self, sigma: torch.Tensor) -> torch.Tensor:
        _, out_scale = self._output_stats_for(sigma)
        return sigma * out_scale
