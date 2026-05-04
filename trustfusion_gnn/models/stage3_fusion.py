"""
Stage 3: Trust refinement and fusion output
Corresponds to:
- Credibility Refinement → τ
- Weighted Aggregation → Ŷ  
- Uncertainty Estimation → σ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class CredibilityRefinement(nn.Module):
    """
    Trust refinement module
    Refine trust scores using temporal information and neighbor consistency
    """
    
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Temporal trust aggregation
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Refinement network
        self.refine_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau_temporal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, N, T, H) node embeddings
            tau_temporal: (batch, N, T) time-varying trust scores
            
        Returns:
            tau_final: (batch, N) final trust scores (one per sensor)
        """
        B, N, T, H = h.shape
        
        # Aggregate temporal information with attention per node
        h_flat = h.view(B * N, T, H)
        h_attended, _ = self.temporal_attention(h_flat, h_flat, h_flat)
        h_pooled = h_attended.mean(dim=1)  # (B*N, H)
        h_pooled = h_pooled.view(B, N, H)
        
        # Statistics of time-varying trust scores
        tau_mean = tau_temporal.mean(dim=-1, keepdim=True)  # (B, N, 1)
        
        # Refine
        combined = torch.cat([h_pooled, tau_mean], dim=-1)
        tau_final = self.refine_net(combined).squeeze(-1)  # (B, N)
        
        return tau_final


class WeightedAggregation(nn.Module):
    """
    Trust-weighted aggregation
    Ŷ = Σ (τ_i / Σ τ_j) · f(h_i)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_sensors: int,
        output_dim: int = 4,
        sensor_to_output: Dict[int, int] = None
    ):
        """
        Args:
            sensor_to_output: dictionary mapping each sensor to an output channel
                             e.g. {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3}
                             means sensors 0 and 2 contribute to output 0 (temperature)
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        
        # Default mapping: assign by index order
        if sensor_to_output is None:
            # Assume: 0,1->temperature, 2,3->humidity, 4,5->soil, 6->light
            sensor_to_output = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3}
        self.register_buffer(
            'sensor_to_output', 
            torch.tensor([sensor_to_output.get(i, 0) for i in range(num_sensors)])
        )
        
        # Per-sensor output projection
        self.sensor_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) node embeddings
            tau: (batch, N) trust scores
            
        Returns:
            y_hat: (batch, T, output_dim) fused output
            weights: (batch, N, output_dim) effective weights used
        """
        B, N, T, H = h.shape
        
        # Sensor output contribution
        sensor_out = self.sensor_proj(h)  # (B, N, T, output_dim)
        
        # Compute weighted fusion for each output channel (avoid in-place assignment)
        y_hat_channels = []
        weight_channels = []
        
        for out_idx in range(self.output_dim):
            # Find sensors contributing to this output
            mask = (self.sensor_to_output == out_idx)  # (N,)
            
            if mask.sum() == 0:
                y_hat_channels.append(torch.zeros(B, T, device=h.device, dtype=h.dtype))
                weight_channels.append(torch.zeros(B, N, device=h.device, dtype=h.dtype))
                continue
                
            # Extract trust scores and outputs of selected sensors
            tau_masked = tau[:, mask]  # (B, num_contributing)
            out_masked = sensor_out[:, mask, :, out_idx]  # (B, num_contributing, T)
            
            # Normalize weights
            w = tau_masked / (tau_masked.sum(dim=1, keepdim=True) + 1e-8)  # (B, num_contributing)
            
            # Weighted aggregation
            weighted_out = (out_masked * w.unsqueeze(-1)).sum(dim=1)  # (B, T)
            y_hat_channels.append(weighted_out)

            # Record weights (map back to N dimensions)
            contributing_idx = torch.where(mask)[0]  # (K,)
            selector = F.one_hot(contributing_idx, num_classes=N).to(dtype=h.dtype, device=h.device)  # (K, N)
            weights_full = torch.matmul(w, selector)  # (B, N)
            weight_channels.append(weights_full)

        y_hat = torch.stack(y_hat_channels, dim=-1)  # (B, T, output_dim)
        weights = torch.stack(weight_channels, dim=-1)  # (B, N, output_dim)
        
        return y_hat, weights


class UncertaintyEstimation(nn.Module):
    """
    Uncertainty estimation
    σ = f(h, τ, var(sensor_outputs))
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + output_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # ensure positive values
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        y_hat: torch.Tensor,
        sensor_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, N, T, H)
            tau: (batch, N)
            y_hat: (batch, T, output_dim) fused output
            sensor_outputs: (batch, N, T, output_dim) per-sensor outputs
            
        Returns:
            sigma: (batch, T, output_dim) uncertainty
        """
        B, N, T, H = h.shape
        output_dim = y_hat.shape[-1]
        
        # Global features
        h_global = h.mean(dim=(1, 2))  # (B, H)
        
        # Mean trust score
        tau_mean = tau.mean(dim=1, keepdim=True)  # (B, 1)
        
        # Variance of sensor outputs (captures inconsistency)
        sensor_var = sensor_outputs.var(dim=1).mean(dim=1)  # (B, output_dim)
        
        # Concatenate features
        combined = torch.cat([h_global, tau_mean, sensor_var], dim=-1)  # (B, H + 1 + output_dim)
        
        # Estimate base uncertainty
        base_sigma = self.mlp(combined)  # (B, output_dim)
        
        # Expand to all time steps
        sigma = base_sigma.unsqueeze(1).expand(-1, T, -1)  # (B, T, output_dim)
        
        return sigma


class AnomalyDetector(nn.Module):
    """
    Anomaly detection module
    Judge sensor anomalies by combining multiple signals
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_criteria: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-criteria anomaly detection
        self.criteria_weights = nn.Parameter(torch.ones(num_criteria) / num_criteria)
        
        # Predict anomaly from embeddings
        self.anomaly_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        raw_input: torch.Tensor,
        neighbor_consistency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) node embeddings
            tau: (batch, N) trust scores
            raw_input: (batch, N, T, F) raw input
            neighbor_consistency: (batch, N) consistency score with neighbors
            
        Returns:
            anomaly_scores: (batch, N) anomaly scores [0, 1]
            anomaly_flags: (batch, N) anomaly flags {0, 1}
        """
        B, N, T, H = h.shape
        
        # Criterion 1: anomaly prediction from embeddings
        h_pooled = h.mean(dim=2)  # (B, N, H)
        score_embedding = self.anomaly_net(h_pooled).squeeze(-1)  # (B, N)
        
        # Criterion 2: low trust score
        score_credibility = 1 - tau  # lower trust => more likely anomalous
        
        # Criterion 3: inconsistency with neighbors
        score_consistency = 1 - neighbor_consistency
        
        # Criterion 4: abnormal temporal change (too large/small change rate)
        diff = torch.diff(raw_input.squeeze(-1), dim=2)
        change_rate = diff.abs().mean(dim=2)  # (B, N)
        # Normalize to [0, 1]
        score_temporal = torch.sigmoid(change_rate - change_rate.mean(dim=1, keepdim=True))
        
        # Weighted combination
        weights = F.softmax(self.criteria_weights, dim=0)
        anomaly_scores = (
            weights[0] * score_embedding +
            weights[1] * score_credibility +
            weights[2] * score_consistency +
            weights[3] * score_temporal
        )
        
        # Binarize
        anomaly_flags = (anomaly_scores > 0.5).float()
        
        return anomaly_scores, anomaly_flags


class Stage3Module(nn.Module):
    """
    Full Stage 3 module
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_sensors: int = 7,
        output_dim: int = 4,
        sensor_to_output: Dict[int, int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.credibility_refiner = CredibilityRefinement(hidden_dim, dropout)
        
        self.aggregator = WeightedAggregation(
            hidden_dim, num_sensors, output_dim, sensor_to_output
        )
        
        self.uncertainty_estimator = UncertaintyEstimation(
            hidden_dim, output_dim, dropout
        )
        
        self.anomaly_detector = AnomalyDetector(hidden_dim, dropout=dropout)
        
        # Sensor output projection (for variance and consistency)
        self.sensor_output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau_temporal: torch.Tensor,
        raw_input: torch.Tensor,
        adj: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H)
            tau_temporal: (batch, N, T)
            raw_input: (batch, N, T, F)
            adj: (N, N)
            
        Returns:
            dict with all outputs
        """
        B, N, T, H = h.shape
        
        # 1. Trust refinement
        tau_final = self.credibility_refiner(h, tau_temporal)
        
        # 2. Weighted aggregation
        y_hat, fusion_weights = self.aggregator(h, tau_final)
        
        # 3. Compute sensor outputs (for uncertainty)
        sensor_outputs = self.sensor_output_proj(h)  # (B, N, T, output_dim)
        
        # 4. Uncertainty estimation
        sigma = self.uncertainty_estimator(h, tau_final, y_hat, sensor_outputs)
        
        # 5. Neighbor consistency
        neighbor_consistency = self._compute_neighbor_consistency(
            raw_input, adj
        )
        
        # 6. Anomaly detection
        anomaly_scores, anomaly_flags = self.anomaly_detector(
            h, tau_final, raw_input, neighbor_consistency
        )
        
        # 7. Overall system confidence
        system_confidence = self._compute_system_confidence(
            tau_final, anomaly_flags
        )
        
        return {
            'y_hat': y_hat,              # (B, T, output_dim)
            'tau': tau_final,            # (B, N)
            'tau_full': tau_temporal,    # (B, N, T)
            'sigma': sigma,              # (B, T, output_dim)
            'anomaly_scores': anomaly_scores,  # (B, N)
            'anomaly_flags': anomaly_flags,    # (B, N)
            'fusion_weights': fusion_weights,  # (B, N, output_dim)
            'system_confidence': system_confidence,  # (B,)
        }
    
    def _compute_neighbor_consistency(
        self,
        raw_input: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency with neighbors"""
        B, N, T, F = raw_input.shape
        
        # Simplification: use temporal mean
        x_mean = raw_input.mean(dim=2).squeeze(-1)  # (B, N)
        
        # Neighbor mean
        neighbor_mean = torch.matmul(adj, x_mean.unsqueeze(-1)).squeeze(-1)  # (B, N)
        
        # Consistency = 1 - normalized difference
        diff = (x_mean - neighbor_mean).abs()
        diff_normalized = diff / (diff.max(dim=1, keepdim=True)[0] + 1e-8)
        consistency = 1 - diff_normalized
        
        return consistency
    
    def _compute_system_confidence(
        self,
        tau: torch.Tensor,
        anomaly_flags: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute overall system confidence
        Consider mean trust score and ratio of anomalous sensors
        """
        # Mean trust score
        avg_credibility = tau.mean(dim=1)  # (B,)
        
        # Ratio of non-anomalous sensors
        healthy_ratio = 1 - anomaly_flags.mean(dim=1)  # (B,)
        
        # Combined system confidence
        system_confidence = 0.6 * avg_credibility + 0.4 * healthy_ratio
        
        return system_confidence