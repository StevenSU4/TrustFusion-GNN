"""
Trainer
"""
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import copy
from datetime import datetime
from pathlib import Path

from config import SystemConfig
from models.trustfusion_gnn import TrustFusionGNN
from losses import TrustFusionLoss, GroundTruth
from metrics import MetricsCalculator, MetricsResult
from graph_builder import GraphBuilder
from data_structures import SystemOutput
from normalization import DataNormalizer


class Trainer:
    """Model trainer"""
    
    def __init__(
        self,
        model: TrustFusionGNN,
        config: SystemConfig,
        device: str = None,
        enable_logging: bool = False,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or config.device
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.run_name = run_name or datetime.now().strftime("train_%Y%m%d_%H%M%S")
        self.logger = self._build_logger() if enable_logging else None
        
        self.model.to(self.device)
        
        # Build graph
        self.graph_builder = GraphBuilder(config)
        self.adj = self.graph_builder.get_combined_adjacency().to(self.device)
        
        # Loss function
        self.criterion = TrustFusionLoss(
            lambda_fusion=config.lambda_fusion,
            lambda_consistency=config.lambda_consistency,
            lambda_credibility=config.lambda_credibility,
            lambda_anomaly=0.6
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning-rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            anomaly_threshold=config.anomaly_threshold
        )

        # Data normalization
        self.normalizer = DataNormalizer(config)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_f1': []
        }
        self.best_val_loss = None

        self._log_event(
            "Trainer initialized",
            extra={
                "device": self.device,
                "run_name": self.run_name,
                "num_sensors": self.config.num_sensors,
                "window_size": self.config.window_size,
            },
        )

    def _build_logger(self) -> logging.Logger:
        """Create a dedicated file logger for one training run."""
        log_dir = self.log_dir or Path("training_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"Trainer.{self.run_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        log_file = log_dir / f"{self.run_name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(file_handler)
        return logger

    def _log_event(self, message: str, extra: Optional[Dict[str, object]] = None):
        """Write one structured training log entry when logging is enabled."""
        if self.logger is None:
            return
        if extra:
            payload = json.dumps(extra, ensure_ascii=True, sort_keys=True)
            self.logger.info("%s | %s", message, payload)
        else:
            self.logger.info(message)

    def _persist_history(self):
        """Persist training history to JSON when logging is enabled."""
        if self.logger is None:
            return
        history_dir = self.log_dir or Path("training_logs")
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{self.run_name}_history.json"
        payload = {
            "run_name": self.run_name,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            X, fusion_target, fault_mask, credibility_target = batch
            
            # Move tensors to device
            X = X.to(self.device)
            fusion_target = fusion_target.to(self.device)
            fault_mask = fault_mask.to(self.device)
            credibility_target = credibility_target.to(self.device)

            X_norm = self.normalizer.normalize_input(X)
            fusion_target_norm = self.normalizer.normalize_output(fusion_target)
            
            # Forward pass
            output = self.model(X_norm, self.adj)
            
            # Build GroundTruth object
            gt = GroundTruth(
                clean_data=X,  # simplified: use input as clean_data
                fusion_target=fusion_target_norm,
                fault_mask=fault_mask,
                fault_types=fault_mask,  # simplified
                credibility_target=credibility_target
            )
            
            # Compute loss
            loss, losses_dict = self.criterion(output, gt, X_norm)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def evaluate(
        self, 
        val_loader: DataLoader
    ) -> Tuple[Dict[str, float], MetricsResult]:
        """Evaluate model"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X, fusion_target, fault_mask, credibility_target = batch
                
                X = X.to(self.device)
                fusion_target = fusion_target.to(self.device)
                fault_mask = fault_mask.to(self.device)
                credibility_target = credibility_target.to(self.device)

                X_norm = self.normalizer.normalize_input(X)
                fusion_target_norm = self.normalizer.normalize_output(fusion_target)
                
                # Forward pass
                output = self.model(X_norm, self.adj)
                
                # Build GroundTruth
                gt = GroundTruth(
                    clean_data=X_norm,
                    fusion_target=fusion_target_norm,
                    fault_mask=fault_mask,
                    fault_types=fault_mask,
                    credibility_target=credibility_target
                )
                
                # Compute loss
                loss, losses_dict = self.criterion(output, gt, X_norm)

                y_hat_denorm = self.normalizer.denormalize_output(output.Y_hat)
                
                # Accumulate losses
                for key, value in losses_dict.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value
                num_batches += 1
                
                # Update metrics
                self.metrics_calculator.update(
                    y_hat=y_hat_denorm,
                    y_true=fusion_target,
                    anomaly_scores=output.anomaly_scores,
                    anomaly_labels=fault_mask.max(dim=-1).values,
                    tau=output.tau,
                    tau_target=credibility_target.mean(dim=-1),
                    system_confidence=output.system_confidence,
                    raw_input=None
                )
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        
        return avg_losses, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        patience: int = None
    ) -> Dict:
        """Full training workflow"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        if patience is None:
            patience = self.config.patience
            
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\nStart training (total {num_epochs} epochs, early-stop patience={patience})")
        print(f"Device: {self.device}")
        print("-" * 60)
        self._log_event(
            "Training started",
            extra={"num_epochs": num_epochs, "patience": patience},
        )
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses, metrics = self.evaluate(val_loader)
            
            # Learning-rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_mae'].append(metrics.mae)
            self.history['val_f1'].append(metrics.anomaly_f1)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_losses['total']:.4f} | "
                  f"Val Loss: {val_losses['total']:.4f} | "
                  f"MAE: {metrics.mae:.4f} | "
                  f"F1: {metrics.anomaly_f1:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            self._log_event(
                "Epoch completed",
                extra={
                    "epoch": epoch + 1,
                    "train_loss": train_losses['total'],
                    "val_loss": val_losses['total'],
                    "mae": metrics.mae,
                    "f1": metrics.anomaly_f1,
                    "epoch_time_sec": epoch_time,
                },
            )
            
            # Early-stop check
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.best_val_loss = best_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  ✓ New best model!")
                self._log_event(
                    "New best model",
                    extra={"epoch": epoch + 1, "best_val_loss": best_val_loss},
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered: no improvement for {patience} epochs.")
                    self._log_event(
                        "Early stopping triggered",
                        extra={"epoch": epoch + 1, "patience": patience},
                    )
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model (val_loss={best_val_loss:.4f})")
            self._log_event(
                "Best model restored",
                extra={"best_val_loss": best_val_loss},
            )

        self._persist_history()
        
        return self.history
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
        self._log_event("Model saved", extra={"path": path})
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")
        self._log_event("Model loaded", extra={"path": path})