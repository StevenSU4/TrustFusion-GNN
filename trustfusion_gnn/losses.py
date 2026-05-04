"""
Loss functions
Corresponding formula: L = ||Ŷ - Y*||² + λ₁L_consistency + λ₂L_credibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GroundTruth:
    """Training labels"""
    clean_data: torch.Tensor           # clean data (B, N, T, F)
    fusion_target: torch.Tensor        # fusion target (B, T, output_F)
    fault_mask: torch.Tensor           # fault mask (B, N, T)
    fault_types: torch.Tensor          # fault types (B, N, T)
    credibility_target: torch.Tensor   # trust target (B, N, T)


class TrustFusionLoss(nn.Module):
    """
    Composite loss function
    L = L_fusion + λ₁·L_consistency + λ₂·L_credibility
    """
    
    def __init__(
        self,
        lambda_fusion: float = 1.0,
        lambda_consistency: float = 0.3,
        lambda_credibility: float = 0.5,
        lambda_anomaly: float = 0.3,
        lambda_uncertainty: float = 0.1
    ):
        super().__init__()
        
        self.lambda_fusion = lambda_fusion
        self.lambda_consistency = lambda_consistency
        self.lambda_credibility = lambda_credibility
        self.lambda_anomaly = lambda_anomaly
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(
        self,
        output,  # SystemOutput
        target: GroundTruth,
        raw_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        """
        losses = {}
        
        # 1. Fusion loss ||Ŷ - Y*||²
        loss_fusion = F.mse_loss(output.Y_hat, target.fusion_target)
        losses['fusion'] = loss_fusion.item()
        
        # 2. Trust-score loss
        loss_credibility = F.binary_cross_entropy(
            output.tau_full,
            target.credibility_target,
            reduction='mean'
        )
        losses['credibility'] = loss_credibility.item()
        
        # 3. Anomaly detection loss
        # Supervise using window-level binary labels: any fault in the window => anomaly
        anomaly_target = target.fault_mask.float().max(dim=-1).values  # (B, N)
        # Positive-class weighting to mitigate low recall caused by sparse anomalies
        pos_ratio = anomaly_target.mean()
        pos_weight = ((1.0 - pos_ratio) / (pos_ratio + 1e-6)).clamp(min=1.0, max=5.0)
        sample_weight = torch.where(
            anomaly_target > 0.5,
            torch.full_like(anomaly_target, pos_weight),
            torch.ones_like(anomaly_target)
        )
        loss_anomaly = F.binary_cross_entropy(
            output.anomaly_scores,
            anomaly_target,
            weight=sample_weight,
            reduction='mean'
        )
        losses['anomaly'] = loss_anomaly.item()
        
        # 4. Spatiotemporal consistency loss
        loss_consistency = self._consistency_loss(
            output.tau, output.anomaly_scores
        )
        losses['consistency'] = loss_consistency.item()
        
        # 5. Uncertainty calibration loss
        loss_uncertainty = self._uncertainty_loss(
            output.Y_hat, target.fusion_target, output.sigma
        )
        losses['uncertainty'] = loss_uncertainty.item()
        
        # Total loss
        total_loss = (
            self.lambda_fusion * loss_fusion +
            self.lambda_credibility * loss_credibility +
            self.lambda_anomaly * loss_anomaly +
            self.lambda_consistency * loss_consistency +
            self.lambda_uncertainty * loss_uncertainty
        )
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def _consistency_loss(
        self,
        tau: torch.Tensor,
        anomaly_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency loss: trust score and anomaly score should be complementary
        τ + anomaly ≈ 1
        """
        consistency = tau + anomaly_scores
        return ((consistency - 1.0) ** 2).mean()
    
    def _uncertainty_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Uncertainty calibration loss
        Negative log-likelihood
        """
        sigma = sigma.clamp(min=1e-4)
        
        nll = 0.5 * (
            torch.log(2 * 3.14159 * sigma ** 2) +
            ((y_hat - y_true) ** 2) / (sigma ** 2)
        )
        
        return nll.mean()