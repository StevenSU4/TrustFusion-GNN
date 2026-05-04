"""
Full TrustFusion-GNN model
Integrates Stage 1, 2, and 3
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .stage1_feature import Stage1Module
from .stage2_graph import Stage2Module
from .stage3_fusion import Stage3Module
from data_structures import SystemInput, SystemOutput


class TrustFusionGNN(nn.Module):
    """
    TrustFusion-GNN: trustworthy data fusion network
    
    Input: X = {X, A, M}
    Output: Ŷ, τ, σ = f_θ(X)
    """
    
    def __init__(
        self,
        num_sensors: int = 7,
        input_dim: int = 1,
        output_dim: int = 4,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        gnn_layers: int = 2,
        num_heads: int = 4,
        statistical_features: int = 8,
        sensor_to_output: Dict[int, int] = None,
        dropout: float = 0.1,
        use_learnable_graph: bool = True
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        
        # Default sensor-to-output mapping
        if sensor_to_output is None:
            # temp_1, temp_2 -> temperature (0)
            # humidity_1, humidity_2 -> humidity (1)
            # soil_1, soil_2 -> soil moisture (2)
            # light -> illumination (3)
            sensor_to_output = {
                0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3
            }
        
        # Stage 1: feature extraction and initial trust estimation
        self.stage1 = Stage1Module(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            statistical_features=statistical_features,
            dropout=dropout
        )
        
        # Stage 2: trust-aware GNN
        self.stage2 = Stage2Module(
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_learnable_graph=use_learnable_graph
        )
        
        # Stage 3: fusion output
        self.stage3 = Stage3Module(
            hidden_dim=hidden_dim,
            num_sensors=num_sensors,
            output_dim=output_dim,
            sensor_to_output=sensor_to_output,
            dropout=dropout
        )
        
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        M: Optional[torch.Tensor] = None
    ) -> SystemOutput:
        """
        Args:
            X: (batch, N, T, F) multi-sensor observation sequence
            A: (N, N) sensor spatial relation graph
            M: (N, D) sensor metadata (optional)
            
        Returns:
            SystemOutput with all outputs
        """
        # Stage 1: feature extraction
        h_temp, s_feat, tau_init = self.stage1(X)
        
        # Stage 2: graph neural network
        h_gnn, tau_temporal, learned_adj, attention = self.stage2(
            h_temp, tau_init, A
        )
        
        # Stage 3: fusion output
        stage3_output = self.stage3(
            h_gnn, tau_temporal, X, learned_adj
        )
        
        # Build output structure
        return SystemOutput(
            Y_hat=stage3_output['y_hat'],
            tau=stage3_output['tau'],
            tau_full=stage3_output['tau_full'],
            anomaly_flags=stage3_output['anomaly_flags'],
            anomaly_scores=stage3_output['anomaly_scores'],
            sigma=stage3_output['sigma'],
            system_confidence=stage3_output['system_confidence'],
            node_embeddings=h_gnn,
            learned_adjacency=learned_adj,
            attention_weights=attention
        )
    
    def get_model_summary(self) -> Dict:
        """Get model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # float32
            'num_sensors': self.num_sensors,
            'output_dim': self.output_dim
        }