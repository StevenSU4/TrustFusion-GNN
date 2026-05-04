"""
Evaluation metrics
Corresponding categories: fusion accuracy, anomaly detection, robustness
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


OUTPUT_CHANNELS = ["temperature", "humidity", "soil_moisture", "light"]


@dataclass
class MetricsResult:
    """Evaluation metric results"""
    # Fusion accuracy
    mae: float
    rmse: float
    mape: float
    per_channel_mae: Dict[str, float]
    per_channel_rmse: Dict[str, float]
    
    # Baseline comparison
    improvement_vs_mean: float
    improvement_vs_median: float
    
    # Anomaly detection
    anomaly_auc: float
    anomaly_precision: float
    anomaly_recall: float
    anomaly_f1: float
    
    # Trust-score evaluation
    credibility_mae: float
    
    # System metrics
    system_confidence_mean: float


class MetricsCalculator:
    """Evaluation metrics calculator"""
    
    def __init__(self, anomaly_threshold: float = 0.5):
        self.anomaly_threshold = anomaly_threshold
        self.reset()
        
    def reset(self):
        """Reset accumulators"""
        self.fusion_preds = []
        self.fusion_targets = []
        self.anomaly_preds = []
        self.anomaly_targets = []
        self.credibility_preds = []
        self.credibility_targets = []
        self.system_confidences = []
        self.raw_inputs = []
        
    def update(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        anomaly_scores: torch.Tensor,
        anomaly_labels: torch.Tensor,
        tau: torch.Tensor,
        tau_target: torch.Tensor,
        system_confidence: torch.Tensor,
        raw_input: torch.Tensor = None
    ):
        """Accumulate one batch of results"""
        self.fusion_preds.append(y_hat.detach().cpu())
        self.fusion_targets.append(y_true.detach().cpu())
        self.anomaly_preds.append(anomaly_scores.detach().cpu())
        self.anomaly_targets.append(anomaly_labels.detach().cpu())
        self.credibility_preds.append(tau.detach().cpu())
        self.credibility_targets.append(tau_target.detach().cpu())
        self.system_confidences.append(system_confidence.detach().cpu())
        if raw_input is not None:
            self.raw_inputs.append(raw_input.detach().cpu())
    
    def compute(self) -> MetricsResult:
        """Compute all metrics"""
        # Concatenate data
        y_hat = torch.cat(self.fusion_preds, dim=0)
        y_true = torch.cat(self.fusion_targets, dim=0)
        anom_pred = torch.cat(self.anomaly_preds, dim=0)
        anom_true = torch.cat(self.anomaly_targets, dim=0)
        cred_pred = torch.cat(self.credibility_preds, dim=0)
        cred_true = torch.cat(self.credibility_targets, dim=0)
        sys_conf = torch.cat(self.system_confidences, dim=0)
        
        # ========== Fusion Accuracy Metrics ==========
        mae = self._mae(y_hat, y_true)
        rmse = self._rmse(y_hat, y_true)
        mape = self._mape(y_hat, y_true)
        per_channel_mae = self._per_channel_metric(y_hat, y_true, self._mae)
        per_channel_rmse = self._per_channel_metric(y_hat, y_true, self._rmse)
        
        # ========== Baseline Comparison ==========
        if self.raw_inputs:
            raw = torch.cat(self.raw_inputs, dim=0)
            baseline_mean = raw.mean(dim=1)  # simple mean
            baseline_median = raw.median(dim=1)[0]  # median
            
            mae_mean = self._mae(baseline_mean, y_true)
            mae_median = self._mae(baseline_median, y_true)
            
            improvement_vs_mean = (mae_mean - mae) / mae_mean * 100
            improvement_vs_median = (mae_median - mae) / mae_median * 100
        else:
            improvement_vs_mean = 0.0
            improvement_vs_median = 0.0
        
        # ========== Anomaly Detection Metrics ==========
        anom_pred_np = anom_pred.numpy().flatten()
        anom_true_np = anom_true.numpy().flatten()
        valid_mask = np.isfinite(anom_pred_np) & np.isfinite(anom_true_np)
        anom_pred_np = anom_pred_np[valid_mask]
        anom_true_np = anom_true_np[valid_mask]
        
        # AUC
        try:
            if anom_true_np.size == 0:
                anomaly_auc = 0.5
            else:
                anom_true_bin = (anom_true_np > 0.5).astype(np.int32)
                if np.unique(anom_true_bin).size < 2:
                    anomaly_auc = 0.5
                else:
                    anomaly_auc = self._binary_auc(anom_true_bin, anom_pred_np)
                    if not np.isfinite(anomaly_auc):
                        anomaly_auc = 0.5
        except Exception:
            anomaly_auc = 0.5
        
        # Precision, Recall, F1
        anom_pred_binary = (anom_pred > self.anomaly_threshold).float()
        precision, recall, f1 = self._precision_recall_f1(
            anom_pred_binary, anom_true > 0.5
        )
        
        # ========== Trust-Score Metrics ==========
        credibility_mae = self._mae(cred_pred, cred_true)
        
        # ========== System Metrics ==========
        system_confidence_mean = sys_conf.mean().item()
        
        return MetricsResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            per_channel_mae=per_channel_mae,
            per_channel_rmse=per_channel_rmse,
            improvement_vs_mean=improvement_vs_mean,
            improvement_vs_median=improvement_vs_median,
            anomaly_auc=anomaly_auc,
            anomaly_precision=precision,
            anomaly_recall=recall,
            anomaly_f1=f1,
            credibility_mae=credibility_mae,
            system_confidence_mean=system_confidence_mean
        )
    
    def _mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return (pred - target).abs().mean().item()
    
    def _rmse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return ((pred - target) ** 2).mean().sqrt().item()
    
    def _mape(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mask = target.abs() > 1e-8
        if mask.sum() == 0:
            return 0.0
        mape = ((pred[mask] - target[mask]).abs() / target[mask].abs()).mean()
        return mape.item() * 100

    def _per_channel_metric(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metric_fn,
    ) -> Dict[str, float]:
        """Compute one metric independently for each output channel."""
        num_channels = pred.shape[-1]
        channel_names = OUTPUT_CHANNELS[:num_channels]
        if len(channel_names) < num_channels:
            channel_names.extend([f"channel_{idx}" for idx in range(len(channel_names), num_channels)])

        channel_metrics: Dict[str, float] = {}
        for idx, name in enumerate(channel_names):
            channel_metrics[name] = metric_fn(pred[..., idx], target[..., idx])
        return channel_metrics
    
    def _precision_recall_f1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[float, float, float]:
        pred = torch.nan_to_num(pred.flatten(), nan=0.0, posinf=1.0, neginf=0.0)
        target = target.flatten().float()
        
        tp = ((pred == 1) & (target == 1)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        
        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1

    def _binary_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Pure NumPy implementation of binary ROC-AUC (with tie-aware average ranks)."""
        y_true = y_true.astype(np.int32)
        y_score = y_score.astype(np.float64)

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = np.argsort(y_score)
        sorted_scores = y_score[order]
        sorted_true = y_true[order]

        ranks = np.zeros_like(sorted_scores, dtype=np.float64)
        i = 0
        n = sorted_scores.shape[0]
        while i < n:
            j = i
            while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0  # rank starts from 1
            ranks[i:j + 1] = avg_rank
            i = j + 1

        sum_ranks_pos = ranks[sorted_true == 1].sum()
        auc_score = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc_score)


def compute_robustness_metrics(
    model,
    test_loader,
    anomaly_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> Dict[float, float]:
    """
    Compute robustness metrics under different anomaly ratios
    """
    robustness = {}
    for ratio in anomaly_ratios:
        pass
    return robustness