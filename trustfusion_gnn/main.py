"""
Main entry point — TrustFusion-GNN architecture and graph demo.

For training from collected sensor data, use train.py.
"""
import torch
import numpy as np
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)


def print_banner():
    print("=" * 70)
    print("  TrustFusion-GNN: Trustworthy Data Fusion Network")
    print("  GNN-Enabled Trustworthy Sensor Fusion for Smart Agriculture")
    print("=" * 70)
    print(f"\nRun time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def demo_model_architecture():
    """Show model architecture and run a forward-pass shape check."""
    from config import get_agricultural_config
    from models.trustfusion_gnn import TrustFusionGNN
    from graph_builder import GraphBuilder

    print("\n" + "=" * 60)
    print("1. Model Architecture")
    print("=" * 60)

    config = get_agricultural_config()

    model = TrustFusionGNN(
        num_sensors=config.num_sensors,
        input_dim=config.input_features,
        output_dim=config.output_features,
        hidden_dim=config.gnn_hidden_dim,
        temporal_layers=config.temporal_layers,
        gnn_layers=config.gnn_layers,
        num_heads=config.num_attention_heads,
        dropout=config.dropout,
    )

    summary = model.get_model_summary()
    print(f"\nModel parameters:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")

    print("\nForward pass shape check:")
    B, N, T, F = 2, config.num_sensors, config.window_size, config.input_features
    X = torch.randn(B, N, T, F)

    graph_builder = GraphBuilder(config)
    A = graph_builder.get_combined_adjacency()

    print(f"  Input X: {X.shape}  (batch={B}, N={N}, T={T}, F={F})")
    print(f"  Adjacency A: {A.shape}")

    with torch.no_grad():
        output = model(X, A)

    print(f"\n  Outputs:")
    print(f"    Ŷ (fused):           {output.Y_hat.shape}  -> (batch, T, output_features)")
    print(f"    τ (trust score):     {output.tau.shape}  -> (batch, N)")
    print(f"    τ_full (time-vary):  {output.tau_full.shape}  -> (batch, N, T)")
    print(f"    σ (uncertainty):     {output.sigma.shape}  -> (batch, T, output_features)")
    print(f"    anomaly_flags:       {output.anomaly_flags.shape}  -> (batch, N)")
    print(f"    anomaly_scores:      {output.anomaly_scores.shape}  -> (batch, N)")
    print(f"    system_confidence:   {output.system_confidence.shape}  -> (batch,)")

    return model, config


def demo_graph_structure():
    """Show sensor graph topology."""
    from config import get_agricultural_config
    from graph_builder import GraphBuilder

    print("\n" + "=" * 60)
    print("2. Sensor Graph Structure")
    print("=" * 60)

    config = get_agricultural_config()
    graph_builder = GraphBuilder(config)

    adj_combined = graph_builder.get_combined_adjacency()

    print("\nSensor list:")
    for i, sid in enumerate(graph_builder.sensor_ids):
        sensor = config.sensors[sid]
        print(f"  {i}: {sid} ({sensor.sensor_type.value}) @ ESP32-{sensor.esp32_id}")

    print(f"\nAdjacency shape: {adj_combined.shape}")
    print(f"\nCombined adjacency (top-left 4x4):\n{adj_combined[:4, :4].numpy().round(3)}")

    return graph_builder


def main():
    print_banner()

    print("Select demo:")
    print("  1. Model architecture")
    print("  2. Sensor graph structure")
    print("  3. Both (default)")

    choice = input("\nEnter choice (1-3, default 3): ").strip() or "3"

    if choice == "1":
        demo_model_architecture()
    elif choice == "2":
        demo_graph_structure()
    else:
        demo_model_architecture()
        demo_graph_structure()

    print("\n" + "=" * 70)
    print("Demo complete.  Run  python train.py  to train on collected data.")
    print("=" * 70)


if __name__ == "__main__":
    main()