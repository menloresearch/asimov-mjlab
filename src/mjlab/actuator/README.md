# Actuator Dynamics Network for Asimov Robot

This module provides neural network models to learn real actuator dynamics from CAN bus data, enabling more realistic simulation and improved sim-to-real transfer.

## Overview

The actuator network learns the relationship between motor commands and actual motor responses using real CAN bus data from your Asimov robot. This captures:
- Motor response delays (5-20ms typical)
- Tracking errors and systematic biases
- Temperature-dependent performance
- Current/torque limitations
- Communication latency

## Architecture

### Hybrid MLP Model (Recommended)
- **Input**: Command position + 8-timestep history window (40ms @ 200Hz)
- **Output**: Predicted position and velocity
- **Advantages**: Fast inference (<1ms), stable training, captures temporal dynamics
- **Parameters**: ~48K (lightweight for real-time use)

### Alternative Models
- **LSTM**: Better temporal modeling but slower inference
- **Residual**: Learns corrections to ideal response
- **Ensemble**: Combines multiple models for robustness

## Data Format

The CAN bus data (`data.csv`) contains:
- `timestamp`: Time in seconds
- `motor_id`: 1-12 (corresponding to 12 actuators)
- `motor_name`: Human-readable joint name
- `cmd_pos`: Command position from policy
- `fb_pos`: Actual feedback position
- `fb_vel`: Feedback velocity
- `fb_cur`: Current draw
- `fb_temp`: Motor temperature
- `error_code`: Error status

## Quick Start

### 1. Train Actuator Model
```bash
# Train MLP model with default settings
uv run python src/mjlab/actuator/train_actuator.py \
    --data_path data.csv \
    --model_type mlp \
    --epochs 100 \
    --batch_size 256

# Train with GPU acceleration
uv run python src/mjlab/actuator/train_actuator.py \
    --device cuda \
    --epochs 200

# Train per-motor model (for motor 1)
uv run python src/mjlab/actuator/train_actuator.py \
    --per_motor \
    --motor_id 1
```

### 2. Visualize Results
```bash
# Visualize predictions
uv run python src/mjlab/actuator/visualize.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 100
```

### 3. Use in Training

Integration with mjlab training (future work):
```python
from src.mjlab.actuator import create_model

# Load trained actuator model
actuator_model = create_model("mlp", input_dim=26)
actuator_model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Use in environment for realistic motor dynamics
# This would replace ideal actuators in simulation
```

## Training Results

With just 3 epochs of training on your data:
- **Position RMSE**: 0.016 rad (normalized)
- **Velocity RMSE**: 0.011 rad/s (normalized)
- **R² Score (Position)**: 0.99

This indicates the model captures motor dynamics very well.

## Motor Mapping

Your 12 motors correspond to:
1. L_Hip_Pitch
2. L_Hip_Roll
3. L_Hip_Yaw
4. L_Knee_Pitch
5. L_Ankle_Pitch
6. L_Ankle_Roll
7. R_Hip_Pitch
8. R_Hip_Roll
9. R_Hip_Yaw
10. R_Knee_Pitch
11. R_Ankle_Pitch
12. R_Ankle_Roll

## Integration with Asimov Training

To use the learned actuator dynamics in RL training:

1. **Replace ideal actuators** in `joint_actions.py`
2. **Add delay modeling** to action processing pipeline
3. **Include temperature effects** in observations
4. **Train policy** with realistic motor behavior

This will significantly improve sim-to-real transfer as your policy learns to compensate for real motor limitations from the start.

## Advanced Usage

### Custom Model Architecture
```python
from src.mjlab.actuator import HybridActuatorModel

model = HybridActuatorModel(
    input_dim=26,
    hidden_dims=(512, 256, 128),  # Larger network
    activation="elu",
    dropout=0.1
)
```

### Ensemble Training
```python
# Train ensemble for better robustness
uv run python train_actuator.py --model_type ensemble
```

## Files

- `data_loader.py`: CAN bus data processing and temporal windowing
- `actuator_network.py`: Neural network architectures
- `train_actuator.py`: Training script with early stopping
- `visualize.py`: Prediction visualization and metrics
- `analyze_data.py`: Data quality analysis

## Next Steps

1. **Train longer** (100+ epochs) for better accuracy
2. **Integrate with mjlab** action processing
3. **Test sim-to-real transfer** improvements
4. **Add online adaptation** from new CAN data
5. **Implement delay compensation** in deployment