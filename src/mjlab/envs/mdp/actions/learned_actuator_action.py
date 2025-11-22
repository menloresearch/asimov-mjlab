"""Learned actuator dynamics action for realistic motor simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.entity import Entity
from mjlab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv
    from mjlab.envs.mdp.actions import actions_config


class LearnedActuatorAction(JointAction):
    """Action term that uses a learned neural network model for actuator dynamics.

    This action term replaces ideal PD control with realistic motor dynamics
    learned from CAN bus data. It handles the 50Hz policy → 200Hz firmware
    execution pattern where commands are held constant for 4 simulation steps.
    """

    def __init__(self, cfg: actions_config.LearnedActuatorActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg=cfg, env=env)

        # Store config
        self.cfg = cfg

        # Load the trained actuator model
        self.actuator_model = self._load_model(cfg.model_path)
        self.actuator_model.to(self.device)
        self.actuator_model.eval()  # Set to evaluation mode

        # Initialize history buffer for temporal windowing
        # Shape: (num_envs, num_joints, window_size, features)
        # Features: [position, velocity, current]
        self.window_size = cfg.window_size
        self.history_buffer = torch.zeros(
            self.num_envs,
            self._num_joints,
            self.window_size,
            3,  # pos, vel, current
            device=self.device
        )

        # Temperature buffer (simplified - could be made dynamic)
        self.temperature = torch.ones(
            self.num_envs,
            self._num_joints,
            1,
            device=self.device
        ) * 25.0  # Default 25°C

        # Step-wise command execution
        self.cmd_hold_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.held_commands = torch.zeros_like(self._processed_actions)

        # Normalization constants (from data loader)
        self.pos_min = -2.0
        self.pos_max = 2.0
        self.vel_max = 40.0
        self.cur_max = 30.0
        self.temp_min = 20.0
        self.temp_max = 50.0

        # For tracking previous state (to compute current/torque proxy)
        self.prev_joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained actuator model."""
        from mjlab.actuator.actuator_network import HybridActuatorModel

        # Check if model exists
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Actuator model not found at {model_path}")

        # Initialize model architecture
        # Input: 1 cmd + 8*3 history + 1 temp = 26 features per motor
        input_dim = 26
        model = HybridActuatorModel(
            input_dim=input_dim,
            hidden_dims=(256, 128, 64),
            output_dim=2,  # position, velocity
            activation="relu"
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded actuator model from {model_path}")
        print(f"Model validation loss: {checkpoint.get('val_loss', 'N/A')}")

        return model

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions with scaling and offset."""
        super().process_actions(actions)

        # Update held commands only at the start of each 20ms cycle (every 4 sim steps)
        # This mimics firmware behavior: policy runs at 50Hz, firmware at 200Hz
        update_mask = (self.cmd_hold_counter == 0)
        self.held_commands[update_mask] = self._processed_actions[update_mask]

        # Increment counter (0, 1, 2, 3, 0, 1, 2, 3, ...)
        self.cmd_hold_counter = (self.cmd_hold_counter + 1) % self.cfg.control_decimation

    def apply_actions(self):
        """Apply learned actuator dynamics with realistic request-response latency.

        The trained model learned from CAN bus data that includes:
        - Command transmission delay (firmware → motor)
        - Motor processing time
        - Response transmission delay (motor → firmware)

        This is the actual latency pattern from the request-response protocol.
        """

        # Get current joint states (these represent the last known feedback)
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        # Estimate current (simplified - proportional to acceleration)
        joint_acc = (joint_vel - self.prev_joint_vel) / self._env.step_dt
        estimated_current = joint_acc * 0.1  # Scaling factor
        self.prev_joint_vel = joint_vel.clone()

        # Update history buffer (shift and add new)
        self.history_buffer = torch.roll(self.history_buffer, shifts=-1, dims=2)
        self.history_buffer[:, :, -1, 0] = joint_pos
        self.history_buffer[:, :, -1, 1] = joint_vel
        self.history_buffer[:, :, -1, 2] = estimated_current

        # Normalize inputs (matching training data normalization)
        cmd_norm = self._normalize_position(self.held_commands)
        pos_hist_norm = self._normalize_position(self.history_buffer[:, :, :, 0])
        vel_hist_norm = self._normalize_velocity(self.history_buffer[:, :, :, 1])
        cur_hist_norm = self._normalize_current(self.history_buffer[:, :, :, 2])
        temp_norm = self._normalize_temperature(self.temperature[:, :, -1:])

        # Prepare input for neural network
        # Shape: (num_envs * num_joints, input_dim)
        batch_size = self.num_envs * self._num_joints

        # Flatten for batch processing
        network_input = torch.cat([
            cmd_norm.view(batch_size, 1),
            pos_hist_norm.view(batch_size, self.window_size),
            vel_hist_norm.view(batch_size, self.window_size),
            cur_hist_norm.view(batch_size, self.window_size),
            temp_norm.view(batch_size, 1)
        ], dim=1)

        # Forward pass through actuator model
        with torch.no_grad():
            predictions = self.actuator_model(network_input)

        # Reshape predictions back to (num_envs, num_joints, 2)
        predictions = predictions.view(self.num_envs, self._num_joints, 2)

        # Denormalize predictions
        predicted_pos = self._denormalize_position(predictions[:, :, 0])
        predicted_vel = self._denormalize_velocity(predictions[:, :, 1])

        if self.cfg.use_position_target:
            # Write predicted positions as targets (standard mode)
            self._asset.write_joint_position_target_to_sim(
                predicted_pos, self._actuator_ids
            )
        else:
            # Alternative: directly set joint states (bypass PD controller)
            # This more directly applies the learned dynamics
            self._asset.data.joint_pos[:, self._joint_ids] = predicted_pos
            self._asset.data.joint_vel[:, self._joint_ids] = predicted_vel

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset action buffers and history for specified environments."""
        super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        # Reset history buffer
        self.history_buffer[env_ids] = 0.0

        # Reset command holding
        self.cmd_hold_counter[env_ids] = 0
        self.held_commands[env_ids] = 0.0

        # Reset temperature to default
        self.temperature[env_ids] = 25.0

        # Reset previous velocity
        self.prev_joint_vel[env_ids] = 0.0

    # Normalization helper methods
    def _normalize_position(self, pos):
        """Normalize position to [-1, 1]."""
        pos_range = self.pos_max - self.pos_min
        return (pos - self.pos_min) / pos_range * 2 - 1

    def _denormalize_position(self, pos_norm):
        """Denormalize position from [-1, 1]."""
        pos_range = self.pos_max - self.pos_min
        return (pos_norm + 1) / 2 * pos_range + self.pos_min

    def _normalize_velocity(self, vel):
        """Normalize velocity to [-1, 1]."""
        return vel / self.vel_max

    def _denormalize_velocity(self, vel_norm):
        """Denormalize velocity from [-1, 1]."""
        return vel_norm * self.vel_max

    def _normalize_current(self, cur):
        """Normalize current to [-1, 1]."""
        return cur / self.cur_max

    def _normalize_temperature(self, temp):
        """Normalize temperature to [-1, 1]."""
        temp_range = self.temp_max - self.temp_min
        return (temp - self.temp_min) / temp_range * 2 - 1