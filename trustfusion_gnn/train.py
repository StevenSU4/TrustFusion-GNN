"""Train TrustFusion-GNN from collected sensor dataset files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from config import get_agricultural_config
from metrics import MetricsResult
from models.trustfusion_gnn import TrustFusionGNN
from dataset_loader import (
    create_dataloader_from_npz,
    evaluate_summary_reasonableness,
    load_json,
)
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train using collected sensor datasets.")
    parser.add_argument("--data-dir", default="collected_data", help="Directory containing .npz dataset files.")
    parser.add_argument("--train-file", default="train.npz", help="Training dataset file name.")
    parser.add_argument("--val-file", default="val.npz", help="Validation dataset file name.")
    parser.add_argument("--train-summary", default="train_summary.json", help="Training summary file name.")
    parser.add_argument("--val-summary", default="val_summary.json", help="Validation summary file name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop patience.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension.")
    parser.add_argument("--output-dir", default="collected_data/training_runs", help="Directory for logs and outputs.")
    return parser.parse_args()


def summarize_reasonableness(data_dir: Path, train_summary_name: str, val_summary_name: str):
    train_summary = load_json(str(data_dir / train_summary_name))
    val_summary = load_json(str(data_dir / val_summary_name))

    train_ok, train_checks = evaluate_summary_reasonableness(train_summary)
    val_ok, val_checks = evaluate_summary_reasonableness(val_summary)

    return {
        "train": {"reasonable": train_ok, "checks": train_checks, "summary": train_summary},
        "val": {"reasonable": val_ok, "checks": val_checks, "summary": val_summary},
        "overall_reasonable": train_ok and val_ok,
    }


def metrics_to_dict(metrics: MetricsResult):
    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "per_channel_mae": metrics.per_channel_mae,
        "per_channel_rmse": metrics.per_channel_rmse,
        "improvement_vs_mean": metrics.improvement_vs_mean,
        "improvement_vs_median": metrics.improvement_vs_median,
        "anomaly_auc": metrics.anomaly_auc,
        "anomaly_precision": metrics.anomaly_precision,
        "anomaly_recall": metrics.anomaly_recall,
        "anomaly_f1": metrics.anomaly_f1,
        "credibility_mae": metrics.credibility_mae,
        "system_confidence_mean": metrics.system_confidence_mean,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reasonableness = summarize_reasonableness(data_dir, args.train_summary, args.val_summary)

    print("Dataset sanity checks:")
    print(json.dumps(reasonableness, indent=2))

    config = get_agricultural_config()
    config.batch_size = args.batch_size

    train_loader = create_dataloader_from_npz(
        str(data_dir / args.train_file), batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader_from_npz(
        str(data_dir / args.val_file), batch_size=args.batch_size, shuffle=False
    )

    model = TrustFusionGNN(
        num_sensors=config.num_sensors,
        input_dim=config.input_features,
        output_dim=config.output_features,
        hidden_dim=args.hidden_dim,
        temporal_layers=1,
        gnn_layers=1,
        num_heads=2,
        dropout=0.1,
    )

    run_name = datetime.now().strftime("training_run_%Y%m%d_%H%M%S")
    trainer = Trainer(
        model,
        config,
        enable_logging=reasonableness["overall_reasonable"],
        log_dir=str(output_dir),
        run_name=run_name,
    )

    trainer.train(train_loader, val_loader, num_epochs=args.epochs, patience=args.patience)
    val_losses, metrics = trainer.evaluate(val_loader)

    print("Validation metrics:")
    print(json.dumps(metrics_to_dict(metrics), indent=2))

    summary = {
        "run_name": run_name,
        "overall_reasonable": reasonableness["overall_reasonable"],
        "reasonableness": reasonableness,
        "validation_losses": val_losses,
        "validation_metrics": metrics_to_dict(metrics),
    }

    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training run completed.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()