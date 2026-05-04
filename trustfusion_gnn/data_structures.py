"""
Data structure definitions
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum
import torch


@dataclass
class SystemInput:
    """
    System input X = {X, A, M}
    Corresponds to the architecture diagram:
    1. Multi-sensor observation sequence X ∈ ℝ^{N×T×F}
    2. Sensor spatial relation graph A ∈ ℝ^{N×N}
    3. Sensor metadata M ∈ ℝ^{N×D}
    """
    # Observation data X ∈ ℝ^{N×T×F}
    X: torch.Tensor                    # shape: (batch, N, T, F) or (N, T, F)
    
    # Adjacency matrix A ∈ ℝ^{N×N}
    A: torch.Tensor                    # shape: (N, N)
    
    # Metadata embedding M ∈ ℝ^{N×D}
    M: Optional[torch.Tensor] = None   # shape: (N, D)
    
    # Additional metadata
    timestamps: Optional[np.ndarray] = None
    sensor_ids: Optional[List[str]] = None

@dataclass
class SystemOutput:
    """System output"""
    Y_hat: torch.Tensor                # (batch, T, F) fused output
    tau: torch.Tensor                  # (batch, N) sensor trust score
    tau_full: torch.Tensor             # (batch, N, T) time-varying trust score
    anomaly_flags: torch.Tensor        # (batch, N) anomaly flags
    anomaly_scores: torch.Tensor       # (batch, N) anomaly scores
    sigma: torch.Tensor                # (batch, T, F) uncertainty
    system_confidence: torch.Tensor    # (batch,) system confidence
    node_embeddings: Optional[torch.Tensor] = None
    learned_adjacency: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None

# ==================== Stage Intermediate Outputs ====================

@dataclass  
class Stage1Output:
    """
    Stage 1 output
    Corresponds to: h_temp, s_feat, τ_init
    """
    h_temp: torch.Tensor      # temporal features (batch, N, T, H)
    s_feat: torch.Tensor      # statistical features (batch, N, S)
    tau_init: torch.Tensor    # initial trust score (batch, N)


@dataclass
class Stage2Output:
    """
    Stage 2 output
    Corresponds to node embeddings after multi-hop message passing
    """
    node_embeddings: torch.Tensor    # (batch, N, T, H)
    attention_weights: torch.Tensor  # (N, N) attention weights
    learned_adj: torch.Tensor        # (N, N) learned adjacency matrix

@dataclass
class FusionResult:
    """Application-layer fusion result"""
    timestamp: datetime
    fused_values: Dict[str, float]
    uncertainties: Dict[str, float]
    sensor_credibility: Dict[str, float]
    anomaly_flags: Dict[str, bool]
    fusion_weights: Dict[str, float]
    system_confidence: float
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)