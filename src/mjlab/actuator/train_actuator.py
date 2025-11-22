#!/usr/bin/env python3
"""Training script for actuator dynamics models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, Optional
from datetime import datetime
import wandb

from mjlab.actuator.data_loader import create_dataloaders
from mjlab.actuator.actuator_network import create_model


class ActuatorTrainer:
    """Trainer class for actuator dynamics models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model: Neural network model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir / "tensorboard")
        self.use_wandb = use_wandb

        # Best model tracking
        self.best_val_loss = float('inf')
        self.early_stop_patience = 30
        self.early_stop_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        pos_loss = 0
        vel_loss = 0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(inputs)

            # Compute losses
            loss = self.criterion(predictions, targets)

            # Separate position and velocity losses for monitoring
            pos_pred, vel_pred = predictions[:, 0], predictions[:, 1]
            pos_target, vel_target = targets[:, 0], targets[:, 1]
            pos_loss_batch = self.criterion(pos_pred, pos_target)
            vel_loss_batch = self.criterion(vel_pred, vel_target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            pos_loss += pos_loss_batch.item()
            vel_loss += vel_loss_batch.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f}")

        # Average losses
        avg_loss = total_loss / num_batches
        avg_pos_loss = pos_loss / num_batches
        avg_vel_loss = vel_loss / num_batches

        return {
            'total_loss': avg_loss,
            'pos_loss': avg_pos_loss,
            'vel_loss': avg_vel_loss,
        }

    def validate(self, loader, split_name: str = "val") -> Dict[str, float]:
        """Validate on a dataset."""
        self.model.eval()
        total_loss = 0
        pos_loss = 0
        vel_loss = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)

                # Compute losses
                loss = self.criterion(predictions, targets)

                # Separate losses
                pos_pred, vel_pred = predictions[:, 0], predictions[:, 1]
                pos_target, vel_target = targets[:, 0], targets[:, 1]
                pos_loss_batch = self.criterion(pos_pred, pos_target)
                vel_loss_batch = self.criterion(vel_pred, vel_target)

                # Accumulate
                total_loss += loss.item()
                pos_loss += pos_loss_batch.item()
                vel_loss += vel_loss_batch.item()
                num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_pos_loss = pos_loss / num_batches
        avg_vel_loss = vel_loss / num_batches

        # Convert to RMSE for interpretability
        rmse = np.sqrt(avg_loss)
        pos_rmse = np.sqrt(avg_pos_loss)
        vel_rmse = np.sqrt(avg_vel_loss)

        return {
            f'{split_name}_loss': avg_loss,
            f'{split_name}_pos_loss': avg_pos_loss,
            f'{split_name}_vel_loss': avg_vel_loss,
            f'{split_name}_rmse': rmse,
            f'{split_name}_pos_rmse': pos_rmse,
            f'{split_name}_vel_rmse': vel_rmse,
        }

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_metrics['total_loss']:.6f}")

            # Validate
            val_metrics = self.validate(self.val_loader, "val")
            print(f"Val Loss: {val_metrics['val_loss']:.6f} "
                  f"(RMSE: {val_metrics['val_rmse']:.6f})")

            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])

            # Logging
            self.log_metrics(epoch, {**train_metrics, **val_metrics})

            # Save checkpoint if best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)
                self.early_stop_counter = 0
                print(f"✓ New best model! Val loss: {self.best_val_loss:.6f}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics['val_loss'], is_best=False)

        # Final test evaluation
        print("\n" + "="*60)
        print("Final Test Evaluation")
        print("="*60)
        test_metrics = self.validate(self.test_loader, "test")
        print(f"Test Loss: {test_metrics['test_loss']:.6f}")
        print(f"Test RMSE: {test_metrics['test_rmse']:.6f}")
        print(f"Position RMSE: {test_metrics['test_pos_rmse']:.6f}")
        print(f"Velocity RMSE: {test_metrics['test_vel_rmse']:.6f}")

        self.writer.close()
        return test_metrics

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb."""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

        # Current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', current_lr, epoch)

        # Weights & Biases
        if self.use_wandb:
            wandb.log({**metrics, 'learning_rate': current_lr, 'epoch': epoch})

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        # Save regular checkpoint
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train actuator dynamics model")
    parser.add_argument('--data_path', type=str, default="/home/menlo/selim/asimov-mjlab/data.csv",
                        help='Path to CAN bus data CSV')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'lstm', 'residual', 'ensemble'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Temporal window size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--per_motor', action='store_true',
                        help='Train separate model per motor')
    parser.add_argument('--motor_id', type=int, default=1,
                        help='Motor ID if training per-motor model')

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="asimov-actuator",
            name=f"{args.model_type}_window{args.window_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        window_size=args.window_size,
        per_motor=args.per_motor,
        motor_id=args.motor_id if args.per_motor else None,
    )

    # Get input/output dimensions
    input_dim = train_loader.dataset.get_input_dim()
    output_dim = train_loader.dataset.get_output_dim()
    print(f"Model input dim: {input_dim}, output dim: {output_dim}")

    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(
        args.model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=(256, 128, 64) if args.model_type in ['mlp', 'residual'] else None,
    )

    # Create trainer
    trainer = ActuatorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
    )

    # Train
    test_metrics = trainer.train(args.epochs)

    # Save final results
    results_path = Path(args.checkpoint_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'test_metrics': test_metrics,
            'model_params': sum(p.numel() for p in model.parameters()),
        }, f, indent=2)

    print(f"\nTraining complete! Results saved to {results_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()