"""ONNX policy inference wrapper for deployment."""

import numpy as np
import torch


class OnnxPolicy:
    """Wrapper for ONNX model inference compatible with RSL-RL policy interface.

    Args:
        onnx_path: Path to the ONNX model file (.onnx)
        device: Device to run inference on (cpu or cuda). Note: ONNX Runtime may use
                CPU even if device='cuda' depending on your installation.
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with:\n"
                "  pip install onnxruntime  # CPU version\n"
                "  pip install onnxruntime-gpu  # GPU version"
            )

        self.device = device

        # Set up ONNX Runtime session
        # Use GPU if available and requested
        providers = ['CPUExecutionProvider']
        if device.startswith('cuda'):
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[INFO]: Loaded ONNX model from {onnx_path}")
        print(f"[INFO]: ONNX Runtime providers: {self.session.get_providers()}")

    def __call__(self, obs: torch.Tensor | dict) -> torch.Tensor:
        """Run inference on observations.

        Args:
            obs: Observation tensor of shape (num_envs, obs_dim) or (obs_dim,)
                 Can also be a dict/TensorDict with "policy" key.

        Returns:
            Action tensor of shape (num_envs, action_dim) or (action_dim,)
        """
        # Handle dict/TensorDict observations
        if isinstance(obs, dict):
            obs = obs.get("policy", obs)
        elif hasattr(obs, "get"):
            # Handle TensorDict-like objects
            obs = obs.get("policy", obs)

        # Store original device to return actions on same device
        original_device = obs.device if isinstance(obs, torch.Tensor) else self.device

        # Convert to numpy
        obs_np = obs.detach().cpu().numpy().astype(np.float32)

        # Ensure batch dimension
        original_shape = obs_np.shape
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        # Run inference
        actions_np = self.session.run(
            [self.output_name],
            {self.input_name: obs_np}
        )[0]

        # Convert back to torch on the original device
        actions = torch.from_numpy(actions_np).to(original_device)

        # Restore original shape
        if len(original_shape) == 1:
            actions = actions.squeeze(0)

        return actions
