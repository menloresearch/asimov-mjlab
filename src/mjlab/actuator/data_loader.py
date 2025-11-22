#!/usr/bin/env python3
"""Data loader for CAN bus actuator data with temporal windowing."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict

# Motor constants - extracted from asimov configuration
# These are approximate values based on the robot specifications

class ActuatorDataset(Dataset):
    """Dataset for actuator dynamics learning with temporal windowing."""

    def __init__(
        self,
        csv_path: str,
        window_size: int = 8,
        mode: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        per_motor: bool = False,
        motor_id: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Initialize the actuator dataset.

        Args:
            csv_path: Path to the CAN bus data CSV file
            window_size: Number of past timesteps to use as input
            mode: "train", "val", or "test"
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            per_motor: If True, create dataset for a single motor
            motor_id: Motor ID if per_motor is True
            normalize: Whether to normalize the data
        """
        self.window_size = window_size
        self.mode = mode
        self.per_motor = per_motor
        self.motor_id = motor_id
        self.normalize = normalize

        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Filter by motor if needed
        if per_motor and motor_id is not None:
            df = df[df['motor_id'] == motor_id]
            print(f"Filtered to motor {motor_id}: {len(df)} samples")

        # Sort by timestamp to ensure temporal order
        df = df.sort_values('timestamp')

        # Extract relevant columns
        self.timestamps = df['timestamp'].values
        self.motor_ids = df['motor_id'].values
        self.cmd_pos = df['cmd_pos'].values
        self.fb_pos = df['fb_pos'].values
        self.fb_vel = df['fb_vel'].values
        self.fb_cur = df['fb_cur'].values
        self.fb_temp = df['fb_temp'].values

        # Get motor limits for normalization
        self.motor_limits = self._get_motor_limits()

        # Normalize if requested
        if self.normalize:
            self._normalize_data()

        # Create temporal windows
        self._create_windows()

        # Split data
        self._split_data(train_split, val_split)

        print(f"Dataset created: {len(self)} samples in {mode} mode")

    def _get_motor_limits(self) -> Dict:
        """Get motor limits from asimov constants."""
        limits = {}

        # Position limits (approximate, based on typical joint ranges)
        limits['pos_min'] = -2.0
        limits['pos_max'] = 2.0

        # Velocity limits (from constants)
        limits['vel_max'] = 40.0  # rad/s (max across all motors)

        # Current limits (approximate)
        limits['cur_max'] = 30.0  # Amps

        # Temperature range
        limits['temp_min'] = 20.0
        limits['temp_max'] = 50.0

        return limits

    def _normalize_data(self):
        """Normalize data to [-1, 1] range."""
        # Position normalization
        pos_range = self.motor_limits['pos_max'] - self.motor_limits['pos_min']
        self.cmd_pos = (self.cmd_pos - self.motor_limits['pos_min']) / pos_range * 2 - 1
        self.fb_pos = (self.fb_pos - self.motor_limits['pos_min']) / pos_range * 2 - 1

        # Velocity normalization
        self.fb_vel = self.fb_vel / self.motor_limits['vel_max']

        # Current normalization
        self.fb_cur = self.fb_cur / self.motor_limits['cur_max']

        # Temperature normalization
        temp_range = self.motor_limits['temp_max'] - self.motor_limits['temp_min']
        self.fb_temp = (self.fb_temp - self.motor_limits['temp_min']) / temp_range * 2 - 1

    def _create_windows(self):
        """Create temporal windows for input/output pairs."""
        self.inputs = []
        self.targets = []

        # For each valid window
        for i in range(self.window_size, len(self.timestamps)):
            # Input features: [cmd_pos(t), history_window]
            # History window: [fb_pos(t-w:t-1), fb_vel(t-w:t-1), fb_cur(t-w:t-1)]

            # Current command
            cmd = self.cmd_pos[i]

            # Historical feedback (window_size timesteps)
            hist_pos = self.fb_pos[i-self.window_size:i]
            hist_vel = self.fb_vel[i-self.window_size:i]
            hist_cur = self.fb_cur[i-self.window_size:i]
            hist_temp = self.fb_temp[i-1:i]  # Just last temperature

            # Combine into input vector
            input_vec = np.concatenate([
                [cmd],                    # 1D: current command
                hist_pos,                 # window_size D: position history
                hist_vel,                 # window_size D: velocity history
                hist_cur,                 # window_size D: current history
                hist_temp,                # 1D: last temperature
            ])

            # Target: next position and velocity
            target_vec = np.array([
                self.fb_pos[i],
                self.fb_vel[i]
            ])

            self.inputs.append(input_vec)
            self.targets.append(target_vec)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        print(f"Created {len(self.inputs)} windows")
        print(f"Input shape: {self.inputs.shape}")
        print(f"Target shape: {self.targets.shape}")

    def _split_data(self, train_split: float, val_split: float):
        """Split data into train/val/test sets."""
        n_samples = len(self.inputs)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        if self.mode == "train":
            self.inputs = self.inputs[:n_train]
            self.targets = self.targets[:n_train]
        elif self.mode == "val":
            self.inputs = self.inputs[n_train:n_train + n_val]
            self.targets = self.targets[n_train:n_train + n_val]
        else:  # test
            self.inputs = self.inputs[n_train + n_val:]
            self.targets = self.targets[n_train + n_val:]

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx])
        )

    def get_input_dim(self) -> int:
        """Get input dimension."""
        return self.inputs.shape[1]

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.targets.shape[1]


def create_dataloaders(
    csv_path: str,
    batch_size: int = 256,
    window_size: int = 8,
    num_workers: int = 4,
    per_motor: bool = False,
    motor_id: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        csv_path: Path to the CAN bus data CSV
        batch_size: Batch size for training
        window_size: Temporal window size
        num_workers: Number of dataloader workers
        per_motor: Whether to create per-motor datasets
        motor_id: Motor ID if per_motor is True

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ActuatorDataset(
        csv_path, window_size, "train", per_motor=per_motor, motor_id=motor_id
    )
    val_dataset = ActuatorDataset(
        csv_path, window_size, "val", per_motor=per_motor, motor_id=motor_id
    )
    test_dataset = ActuatorDataset(
        csv_path, window_size, "test", per_motor=per_motor, motor_id=motor_id
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    csv_path = "/home/menlo/selim/asimov-mjlab/data.csv"

    print("Testing dataloader creation...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path,
        batch_size=256,
        window_size=8,
        per_motor=False  # Use all motors
    )

    print(f"\nDataloader stats:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    inputs, targets = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Input: {inputs.shape}")
    print(f"Target: {targets.shape}")

    # Get dimensions for model creation
    print(f"\nModel dimensions:")
    print(f"Input dim: {train_loader.dataset.get_input_dim()}")
    print(f"Output dim: {train_loader.dataset.get_output_dim()}")