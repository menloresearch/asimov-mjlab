#!/usr/bin/env python3
"""Visualization script for actuator model predictions."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

from data_loader import create_dataloaders
from actuator_network import create_model

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Using text-based visualization.")


def load_trained_model(checkpoint_path: str, model_type: str, input_dim: int, device: str = "cpu"):
    """Load a trained model from checkpoint."""
    model = create_model(model_type, input_dim=input_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Validation loss at checkpoint: {checkpoint['val_loss']:.6f}")
    return model


def evaluate_predictions(model, dataloader, device: str = "cpu", num_samples: int = 100):
    """Evaluate model predictions on a dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)[:num_samples]
    targets = np.concatenate(all_targets, axis=0)[:num_samples]
    inputs = np.concatenate(all_inputs, axis=0)[:num_samples]

    return predictions, targets, inputs


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Position and velocity
    pos_pred, vel_pred = predictions[:, 0], predictions[:, 1]
    pos_target, vel_target = targets[:, 0], targets[:, 1]

    # MSE
    pos_mse = np.mean((pos_pred - pos_target) ** 2)
    vel_mse = np.mean((vel_pred - vel_target) ** 2)
    total_mse = np.mean((predictions - targets) ** 2)

    # RMSE
    pos_rmse = np.sqrt(pos_mse)
    vel_rmse = np.sqrt(vel_mse)
    total_rmse = np.sqrt(total_mse)

    # MAE
    pos_mae = np.mean(np.abs(pos_pred - pos_target))
    vel_mae = np.mean(np.abs(vel_pred - vel_target))

    # R² score
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    pos_r2 = r2_score(pos_target, pos_pred)
    vel_r2 = r2_score(vel_target, vel_pred)

    return {
        'pos_mse': pos_mse,
        'vel_mse': vel_mse,
        'total_mse': total_mse,
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse,
        'total_rmse': total_rmse,
        'pos_mae': pos_mae,
        'vel_mae': vel_mae,
        'pos_r2': pos_r2,
        'vel_r2': vel_r2,
    }


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, inputs: np.ndarray,
                     save_path: str = None, num_plots: int = 4):
    """Plot predicted vs actual trajectories."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available. Showing text-based comparison:")
        print_text_comparison(predictions, targets, num_samples=10)
        return

    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 3 * num_plots))
    if num_plots == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_plots, len(predictions))):
        # Position subplot
        ax_pos = axes[i, 0]
        ax_pos.plot(targets[i, 0], 'b-', label='Actual Position', linewidth=2)
        ax_pos.plot(predictions[i, 0], 'r--', label='Predicted Position', linewidth=2)
        ax_pos.set_xlabel('Time Step')
        ax_pos.set_ylabel('Position (normalized)')
        ax_pos.set_title(f'Sample {i+1}: Position Tracking')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)

        # Velocity subplot
        ax_vel = axes[i, 1]
        ax_vel.plot(targets[i, 1], 'b-', label='Actual Velocity', linewidth=2)
        ax_vel.plot(predictions[i, 1], 'r--', label='Predicted Velocity', linewidth=2)
        ax_vel.set_xlabel('Time Step')
        ax_vel.set_ylabel('Velocity (normalized)')
        ax_vel.set_title(f'Sample {i+1}: Velocity Tracking')
        ax_vel.legend()
        ax_vel.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_error_distribution(predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """Plot error distribution histograms."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nError statistics (text-based):")
        print_error_statistics(predictions, targets)
        return

    pos_errors = predictions[:, 0] - targets[:, 0]
    vel_errors = predictions[:, 1] - targets[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Position error histogram
    axes[0].hist(pos_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Position Error (normalized)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Position Error Distribution\nMean: {np.mean(pos_errors):.4f}, Std: {np.std(pos_errors):.4f}')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].legend()

    # Velocity error histogram
    axes[1].hist(vel_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Velocity Error (normalized)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Velocity Error Distribution\nMean: {np.mean(vel_errors):.4f}, Std: {np.std(vel_errors):.4f}')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")

    plt.show()


def print_text_comparison(predictions: np.ndarray, targets: np.ndarray, num_samples: int = 10):
    """Print text-based comparison of predictions vs targets."""
    print("\n" + "="*60)
    print("Sample Predictions vs Targets (normalized values)")
    print("="*60)
    print(f"{'Sample':<8} {'Pred Pos':<12} {'True Pos':<12} {'Pos Error':<12}")
    print(f"{'':8} {'Pred Vel':<12} {'True Vel':<12} {'Vel Error':<12}")
    print("-"*60)

    for i in range(min(num_samples, len(predictions))):
        pos_pred, vel_pred = predictions[i]
        pos_true, vel_true = targets[i]
        pos_err = pos_pred - pos_true
        vel_err = vel_pred - vel_true

        print(f"{i+1:<8} {pos_pred:>11.4f} {pos_true:>11.4f} {pos_err:>11.4f}")
        print(f"{'':8} {vel_pred:>11.4f} {vel_true:>11.4f} {vel_err:>11.4f}")
        print("-"*60)


def print_error_statistics(predictions: np.ndarray, targets: np.ndarray):
    """Print error statistics in text format."""
    pos_errors = predictions[:, 0] - targets[:, 0]
    vel_errors = predictions[:, 1] - targets[:, 1]

    print("\n" + "="*60)
    print("Error Statistics")
    print("="*60)

    print("\nPosition Errors:")
    print(f"  Mean:     {np.mean(pos_errors):>10.6f}")
    print(f"  Std:      {np.std(pos_errors):>10.6f}")
    print(f"  Min:      {np.min(pos_errors):>10.6f}")
    print(f"  Max:      {np.max(pos_errors):>10.6f}")
    print(f"  Median:   {np.median(pos_errors):>10.6f}")

    print("\nVelocity Errors:")
    print(f"  Mean:     {np.mean(vel_errors):>10.6f}")
    print(f"  Std:      {np.std(vel_errors):>10.6f}")
    print(f"  Min:      {np.min(vel_errors):>10.6f}")
    print(f"  Max:      {np.max(vel_errors):>10.6f}")
    print(f"  Median:   {np.median(vel_errors):>10.6f}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize actuator model predictions")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default="/home/menlo/selim/asimov-mjlab/data.csv",
                        help='Path to CAN bus data CSV')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'lstm', 'residual', 'ensemble'],
                        help='Type of model')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Temporal window size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save plots')
    parser.add_argument('--per_motor', action='store_true',
                        help='Use per-motor model')
    parser.add_argument('--motor_id', type=int, default=1,
                        help='Motor ID if using per-motor model')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        window_size=args.window_size,
        per_motor=args.per_motor,
        motor_id=args.motor_id if args.per_motor else None,
    )

    # Select appropriate dataloader
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    # Get input dimension
    input_dim = dataloader.dataset.get_input_dim()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_trained_model(args.checkpoint, args.model_type, input_dim, args.device)

    # Evaluate predictions
    print(f"Evaluating on {args.split} set...")
    predictions, targets, inputs = evaluate_predictions(
        model, dataloader, args.device, args.num_samples
    )

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)

    # Print metrics
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:15s}: {value:>10.6f}")

    # Visualize predictions
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        plot_predictions(
            predictions, targets, inputs,
            save_path=save_dir / f"{args.model_type}_{args.split}_predictions.png",
            num_plots=4
        )
        plot_error_distribution(
            predictions, targets,
            save_path=save_dir / f"{args.model_type}_{args.split}_errors.png"
        )
    else:
        print_text_comparison(predictions, targets)
        print_error_statistics(predictions, targets)

    print(f"\nVisualization complete! Results saved to {save_dir}")


if __name__ == "__main__":
    main()