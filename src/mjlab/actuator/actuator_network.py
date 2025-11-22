#!/usr/bin/env python3
"""Neural network models for actuator dynamics prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class HybridActuatorModel(nn.Module):
    """
    MLP model with temporal windowing for actuator dynamics.

    This model takes a temporal window of past states and the current command
    to predict the next motor state (position and velocity).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        output_dim: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize the hybrid actuator model.

        Args:
            input_dim: Input dimension (cmd + window_size * features)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (typically 2: pos, vel)
            activation: Activation function ("relu", "elu", "tanh")
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU/ELU, Xavier for tanh
                if isinstance(self.activation, (nn.ReLU, nn.ELU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim] (predicted pos, vel)
        """
        return self.mlp(x)


class LightweightLSTMModel(nn.Module):
    """
    Lightweight LSTM model for actuator dynamics.

    Uses LSTM to process temporal sequences and predict motor states.
    """

    def __init__(
        self,
        input_dim: int = 4,  # cmd_pos, fb_pos, fb_vel, fb_cur at each timestep
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize the LSTM model.

        Args:
            input_dim: Input features per timestep
            hidden_dim: LSTM hidden dimension
            output_dim: Output dimension (pos, vel)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            hidden: Optional initial hidden state

        Returns:
            output: Predictions [batch_size, output_dim]
            hidden: Final hidden state
        """
        # Process through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]

        # Project to output dimension
        output = self.output_layer(last_output)

        return output, hidden


class EnsembleActuatorModel(nn.Module):
    """
    Ensemble of actuator models for improved robustness.

    Combines predictions from multiple models (MLP, LSTM) for better accuracy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        lstm_hidden: int = 64,
        output_dim: int = 2,
        use_lstm: bool = False,
    ):
        """
        Initialize ensemble model.

        Args:
            input_dim: Input dimension for MLP
            hidden_dims: Hidden dimensions for MLP
            lstm_hidden: Hidden dimension for LSTM
            output_dim: Output dimension
            use_lstm: Whether to include LSTM in ensemble
        """
        super().__init__()

        # MLP model
        self.mlp = HybridActuatorModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        # Optional LSTM model
        self.use_lstm = use_lstm
        if use_lstm:
            # LSTM expects different input format
            self.lstm = LightweightLSTMModel(
                input_dim=4,  # Simplified input
                hidden_dim=lstm_hidden,
                output_dim=output_dim,
            )

        # Learned weighting for ensemble
        self.weight_mlp = nn.Parameter(torch.tensor(0.7))
        self.weight_lstm = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor, x_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining multiple models.

        Args:
            x: Input for MLP [batch_size, input_dim]
            x_seq: Optional sequential input for LSTM [batch_size, seq_len, 4]

        Returns:
            Weighted average of model predictions
        """
        # MLP prediction
        pred_mlp = self.mlp(x)

        if self.use_lstm and x_seq is not None:
            # LSTM prediction
            pred_lstm, _ = self.lstm(x_seq)

            # Weighted combination
            weights = F.softmax(torch.stack([self.weight_mlp, self.weight_lstm]), dim=0)
            return weights[0] * pred_mlp + weights[1] * pred_lstm
        else:
            return pred_mlp


class ActuatorResidualModel(nn.Module):
    """
    Residual learning model for actuator error compensation.

    Learns the residual between ideal response and actual response.
    Useful for sim-to-real transfer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        output_dim: int = 2,
    ):
        """
        Initialize residual model.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (residual corrections)
        """
        super().__init__()

        # Base prediction (simple linear model for ideal response)
        self.base_model = nn.Linear(1, output_dim)  # cmd_pos -> ideal response

        # Residual network (learns corrections)
        self.residual_net = HybridActuatorModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation="tanh",  # Bounded corrections
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual learning.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Corrected predictions [batch_size, output_dim]
        """
        # Extract command position (first element)
        cmd_pos = x[:, 0:1]

        # Base prediction (ideal response)
        base_pred = self.base_model(cmd_pos)

        # Residual correction
        residual = self.residual_net(x)

        # Add residual to base
        return base_pred + residual


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create actuator models.

    Args:
        model_type: Type of model ("mlp", "lstm", "ensemble", "residual")
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    if model_type == "mlp":
        return HybridActuatorModel(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == "lstm":
        return LightweightLSTMModel(
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == "ensemble":
        return EnsembleActuatorModel(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == "residual":
        return ActuatorResidualModel(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing actuator network models...")

    # Test dimensions (from dataloader)
    input_dim = 26  # 1 cmd + 8*3 history + 1 temp
    output_dim = 2  # pos, vel
    batch_size = 32

    # Test MLP model
    print("\n1. Testing Hybrid MLP Model:")
    mlp_model = create_model("mlp", input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    y = mlp_model(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")

    # Test LSTM model
    print("\n2. Testing Lightweight LSTM Model:")
    lstm_model = create_model("lstm", input_dim=4, output_dim=output_dim)
    x_seq = torch.randn(batch_size, 8, 4)  # 8 timesteps, 4 features
    y, hidden = lstm_model(x_seq)
    print(f"   Input: {x_seq.shape} -> Output: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # Test Residual model
    print("\n3. Testing Residual Model:")
    res_model = create_model("residual", input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    y = res_model(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in res_model.parameters()):,}")

    print("\nAll models created successfully!")