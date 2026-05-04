"""
Stage 2: Trust-aware GNN information fusion
Corresponds to:
- Credibility-Aware Graph Attention: α_ij = softmax(τ_i · τ_j · attention(h_i, h_j))
- Multi-hop Message Passing: h_i^{(l+1)} = Update(h_i^{(l)}, Σ α_ij · h_j^{(l)})
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CredibilityAwareGraphAttention(nn.Module):
    """
    Trust-aware graph attention
    α_ij = softmax(τ_i · τ_j · attention(h_i, h_j))
    Highly trusted nodes receive larger attention weights
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Trust modulation
        self.credibility_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, H) node features
            tau: (batch, N) trust scores
            adj_mask: (N, N) adjacency mask
            
        Returns:
            h_out: (batch, N, H) updated node features
            attention: (batch, num_heads, N, N) attention weights
        """
        B, N, H = h.shape
        
        # Compute Q, K, V
        Q = self.W_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (B, num_heads, N, N)
        
        # Trust modulation: τ_i · τ_j
        tau_i = tau.unsqueeze(-1)  # (B, N, 1)
        tau_j = tau.unsqueeze(-2)  # (B, 1, N)
        credibility_weight = tau_i * tau_j  # (B, N, N)
        
        # Project to multi-head space
        cred_bias = self.credibility_proj(credibility_weight.unsqueeze(-1))  # (B, N, N, num_heads)
        cred_bias = cred_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
        
        # Combine attention scores
        attn_scores = attn_scores + cred_bias
        
        # Apply adjacency mask
        adj_mask_expanded = adj_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn_scores = attn_scores.masked_fill(adj_mask_expanded == 0, float('-inf'))
        
        # Softmax
        attention = F.softmax(attn_scores, dim=-1)
        attention = self.dropout(attention)
        
        # Aggregate
        h_out = torch.matmul(attention, V)  # (B, num_heads, N, head_dim)
        h_out = h_out.transpose(1, 2).contiguous().view(B, N, H)
        h_out = self.output_proj(h_out)
        
        return h_out, attention.mean(dim=1)  # return mean attention


class MultiHopMessagePassing(nn.Module):
    """
    Multi-hop message passing
    h_i^{(l+1)} = Update(h_i^{(l)}, Σ α_ij · h_j^{(l)})
    Multi-hop propagation helps trusted information cover anomalous nodes
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Trust-aware attention
        self.attention = CredibilityAwareGraphAttention(
            hidden_dim, num_heads, dropout
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, H)
            tau: (batch, N)
            adj_mask: (N, N)
        Returns:
            h_out: (batch, N, H)
            attention: (batch, N, N)
        """
        # Attention aggregation
        h_attn, attention = self.attention(h, tau, adj_mask)
        
        # Update: combine self and neighbor information
        h_combined = torch.cat([h, h_attn], dim=-1)
        h_update = self.update_mlp(h_combined)
        
        # Residual + LayerNorm
        h = self.norm1(h + h_update)
        
        # FFN
        h = self.norm2(h + self.ffn(h))
        
        return h, attention


class Stage2Module(nn.Module):
    """
    Full Stage 2 module
    Multi-layer trust-aware graph neural network
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_learnable_graph: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_learnable_graph = use_learnable_graph
        
        # Multi-layer message passing
        self.layers = nn.ModuleList([
            MultiHopMessagePassing(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Learnable graph (if enabled)
        if use_learnable_graph:
            self.graph_learner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Trust-score updater
        self.tau_updater = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) temporal node features
            tau: (batch, N) initial trust score
            adj: (N, N) base adjacency matrix
            
        Returns:
            h_out: (batch, N, T, H) updated node features
            tau_refined: (batch, N, T) refined time-varying trust score
            learned_adj: (N, N) learned adjacency matrix
            attention_weights: (batch, N, N) final attention weights
        """
        B, N, T, H = h.shape
        
        # Learn graph structure
        learned_adj = adj
        if self.use_learnable_graph:
            # Learn graph from time-averaged features
            h_mean = h.mean(dim=2)  # (B, N, H)
            h_proj = self.graph_learner(h_mean)  # (B, N, H)
            
            # Compute similarity
            h_norm = F.normalize(h_proj, dim=-1)
            sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, N, N)
            sim = sim.mean(dim=0)  # (N, N) batch average
            
            # Mix with base graph
            learned_adj = 0.7 * adj + 0.3 * torch.softmax(sim, dim=-1)
        
        # Store time-varying trust scores
        h_out_list = []
        tau_list = []
        attention_weights = None
        
        # Process each time step
        for t in range(T):
            h_t = h[:, :, t, :]  # (B, N, H)
            
            # Multi-layer message passing
            tau_t = tau
            for layer in self.layers:
                h_t, attention_weights = layer(h_t, tau_t, learned_adj)
                # Update trust scores
                tau_t = self.tau_updater(h_t).squeeze(-1)  # (B, N)
            
            h_out_list.append(h_t)
            tau_list.append(tau_t)
        
        # Stack temporal output features
        h_out = torch.stack(h_out_list, dim=2)  # (B, N, T, H)

        # Stack time-varying trust scores
        tau_refined = torch.stack(tau_list, dim=2)  # (B, N, T)
        
        return h_out, tau_refined, learned_adj, attention_weights