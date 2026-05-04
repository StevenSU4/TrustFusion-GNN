"""
Graph construction module
Build sensor spatial relation graph A ∈ ℝ^{N×N}
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SystemConfig, SensorMeta, SensorType


class GraphBuilder:
    """
    Graph builder
    Build adjacency matrices based on physical distance and sensor types
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.sensor_ids = list(config.sensors.keys())
        self.num_sensors = len(self.sensor_ids)
        
        # Precompute and cache graph variants
        self._distance_adj = None
        self._type_adj = None
        self._correlation_adj = None
        self._combined_adj = None
        
    def get_distance_adjacency(self) -> torch.Tensor:
        """
        Build adjacency from physical distances
        Using Gaussian kernel: A_ij = exp(-d_ij^2 / (2*sigma^2))
        """
        if self._distance_adj is not None:
            return self._distance_adj
            
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # Get positions
        positions = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            positions.append(sensor.position)
        positions = np.array(positions)  # (N, 3)
        
        # Compute distance matrix
        for i in range(N):
            for j in range(N):
                dist = np.linalg.norm(positions[i] - positions[j])
                # Gaussian kernel
                sigma = 3.0  # tunable hyperparameter
                adj[i, j] = np.exp(-dist**2 / (2 * sigma**2))
        
        # Normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self._distance_adj = adj
        return adj
    
    def get_type_adjacency(self) -> torch.Tensor:
        """
        Build adjacency from sensor types
        Sensors of the same type have stronger links
        """
        if self._type_adj is not None:
            return self._type_adj
            
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # Get sensor types
        types = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            types.append(sensor.sensor_type)
        
        # Strong links for same-type sensors
        for i in range(N):
            for j in range(N):
                if types[i] == types[j]:
                    adj[i, j] = 1.0
                else:
                    # Weak links across different types (physical correlation)
                    adj[i, j] = 0.2
        
        # Normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self._type_adj = adj
        return adj
    
    def get_esp32_adjacency(self) -> torch.Tensor:
        """
        Build adjacency by ESP32 group
        Sensors on the same ESP32 have stronger links (possibly shared failure modes)
        """
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # Get ESP32 groups
        esp32_ids = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            esp32_ids.append(sensor.esp32_id)
        
        for i in range(N):
            for j in range(N):
                if esp32_ids[i] == esp32_ids[j]:
                    adj[i, j] = 1.0
                else:
                    adj[i, j] = 0.3
        
        # Normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        return adj
    
    def get_combined_adjacency(
        self,
        distance_weight: float = 0.4,
        type_weight: float = 0.4,
        esp32_weight: float = 0.2
    ) -> torch.Tensor:
        """
        Combine multiple adjacency matrices
        """
        if self._combined_adj is not None:
            return self._combined_adj
            
        adj_dist = self.get_distance_adjacency()
        adj_type = self.get_type_adjacency()
        adj_esp32 = self.get_esp32_adjacency()
        
        combined = (
            distance_weight * adj_dist +
            type_weight * adj_type +
            esp32_weight * adj_esp32
        )
        
        # Normalize
        combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
        
        # Add self-loops
        combined = 0.5 * combined + 0.5 * torch.eye(self.num_sensors)
        
        self._combined_adj = combined
        return combined
    
    def get_k_nearest_neighbors(self, k: int = 3) -> torch.Tensor:
        """
        Distance-based KNN graph
        """
        adj = self.get_distance_adjacency()
        
        # Keep top-k connections per node
        _, indices = adj.topk(k + 1, dim=1)  # +1 includes self
        
        mask = torch.zeros_like(adj)
        for i in range(self.num_sensors):
            mask[i, indices[i]] = 1.0
        
        # Symmetrize
        mask = (mask + mask.T) / 2
        mask = (mask > 0).float()
        
        # Apply mask
        adj_knn = adj * mask
        
        # Normalize
        adj_knn = adj_knn / (adj_knn.sum(dim=1, keepdim=True) + 1e-8)
        
        return adj_knn
    
    def visualize_graph(self):
        """Visualize graph structure (optional)"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            adj = self.get_combined_adjacency().numpy()
            
            G = nx.from_numpy_array(adj)
            
            # Set node labels
            labels = {i: self.sensor_ids[i] for i in range(self.num_sensors)}
            
            # Set node colors by sensor type
            colors = []
            color_map = {
                SensorType.TEMPERATURE: 'red',
                SensorType.HUMIDITY: 'blue',
                SensorType.SOIL_MOISTURE: 'brown',
                SensorType.LIGHT: 'yellow'
            }
            for sid in self.sensor_ids:
                sensor = self.config.sensors[sid]
                colors.append(color_map.get(sensor.sensor_type, 'gray'))
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(
                G, pos,
                labels=labels,
                node_color=colors,
                node_size=1000,
                font_size=8,
                width=[adj[i, j] * 3 for i, j in G.edges()]
            )
            plt.title("Sensor Graph Structure")
            plt.savefig("sensor_graph.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Graph visualization saved to sensor_graph.png")
            
        except ImportError:
            print("Please install matplotlib and networkx for graph visualization")