# ASIMOV Robot Project Notes

## CRITICAL: ONLY USE ASIMOV! NEVER USE UNITREE/G1/GO1!
- **This project is EXCLUSIVELY for the ASIMOV bipedal robot**
- **NEVER test with Unitree robots (G1, GO1, etc.)**
- **NEVER reference or copy G1/GO1 configurations**
- **ALWAYS use Asimov for all testing and development**

## Asimov Robot Specifications
- 12 motors total (6 per leg)
- Motors: Hip pitch/roll/yaw, Knee pitch, Ankle pitch/roll
- Narrow stance: 11.3 cm hip width
- Canted hip pitch design
- Limited ankle ROM: ±20° pitch, ±15° roll

## CAN Bus Data and Actuator Dynamics
- Firmware runs at 200Hz
- Policy runs at 50Hz
- No interpolation between commands - held constant for 4 timesteps
- CAN data captures real request-response protocol latency
- **Trained MLP model achieves 0.008322 rad RMSE (0.48 degrees)** - EXCELLENT!

## Actuator Network Training Results
- Training completed successfully at epoch 82 (early stopped)
- Final validation loss: 0.000069
- Final RMSE: **0.008322 rad (0.48 degrees)**
- Model saved to: `actuator_checkpoints/best_model.pth`
- Model has 48,194 parameters
- Training used 61,130 samples, validation 13,099 samples

## Key Files
- `/plot_action.py` - Contains motor ID to joint name mapping
- `/src/mjlab/actuator/` - Actuator network training code
- `/src/mjlab/tasks/velocity/config/asimov/` - Asimov environment configurations
- `/src/mjlab/envs/mdp/actions/learned_actuator_action.py` - Learned actuator integration
- `actuator_checkpoints/best_model.pth` - Trained actuator model

## Environment Registration
- `Mjlab-Velocity-Flat-Asimov` - Baseline with ideal actuators
- `Mjlab-Velocity-Flat-Asimov-Learned` - With learned actuator dynamics
- `Mjlab-Velocity-Rough-Asimov` - Rough terrain baseline
- `Mjlab-Velocity-Rough-Asimov-Learned` - Rough terrain with learned dynamics

## IMPORTANT DEVELOPMENT NOTES
- **ALWAYS use `uv run` instead of `python` directly!**
- **ALWAYS use `uv add` for dependencies, not pip!**
- This project uses uv for dependency management

## HARDWARE - SUPER POWERFUL GPUs! 🚀
- **2x NVIDIA RTX A6000 GPUs**
- **48 GB VRAM each** (96 GB total!)
- **Compute capability 8.6** (Ampere architecture)
- Can easily handle 4096+ environments for training
- Use large batch sizes and many parallel environments!

## Current Status - SUCCESS! 🎉
- ✅ Actuator network trained successfully (82 epochs, 0.008322 rad RMSE)
- ✅ Learned actuator action integrated into environment
- ✅ Model checkpoint verified at `actuator_checkpoints/best_model.pth`
- ✅ Terrain sensor issue FIXED (removed secondary filter)
- ✅ Baseline Asimov training WORKS
- ✅ Learned actuator Asimov training WORKS with realistic motor dynamics!

## How to Train - Simple Commands

### 🚀 Quick Start - Just Run This!
```bash
# Train Asimov with learned actuator dynamics (RECOMMENDED)
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Asimov-Learned --env.scene.num_envs 4096

# Train on rough terrain
MUJOCO_GL=egl uv run train Mjlab-Velocity-Rough-Asimov-Learned --env.scene.num_envs 4096

# Train baseline (ideal actuators) for comparison
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Asimov --env.scene.num_envs 4096
```

### 📊 Visualize Your Trained Policy

**IMPORTANT**: Use `--video True` (not just `--video`) and `--checkpoint-file` with full path!

```bash
# View your trained Asimov robot in action (NO VIDEO)
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
  --checkpoint-file logs/rsl_rl/asimov_velocity/2025-11-22_13-35-13/model_450.pt

# Record a video (IMPORTANT: --video True not just --video)
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
  --checkpoint-file logs/rsl_rl/asimov_velocity/2025-11-22_13-35-13/model_450.pt \
  --video True

# Record video with custom length and single environment
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
  --checkpoint-file logs/rsl_rl/asimov_velocity/2025-11-22_13-35-13/model_450.pt \
  --video True \
  --video-length 500 \
  --num-envs 1

# Visualize with fewer environments (cleaner view)
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
  --checkpoint-file logs/rsl_rl/asimov_velocity/2025-11-22_13-35-13/model_450.pt \
  --num-envs 1
```

**Where to find checkpoints:**
- Checkpoints are in: `logs/rsl_rl/asimov_velocity/<timestamp>/`
- Model files: `model_0.pt`, `model_50.pt`, `model_100.pt`, etc.
- Videos saved to: `logs/rsl_rl/asimov_velocity/<timestamp>/videos/play/`

**Example checkpoint paths:**
```bash
# Find your latest training run
ls -lt logs/rsl_rl/asimov_velocity/

# List checkpoints in a run
ls -lh logs/rsl_rl/asimov_velocity/2025-11-22_13-35-13/
```

### Performance Scaling Guide
| GPU Memory | Environments | Mini-batches | Expected FPS |
|------------|-------------|--------------|--------------|
| 24GB (4090) | 4096 | 4 | ~125K |
| 48GB (A6000) | 8192 | 8 | ~200K+ |
| 48GB (A6000) | 12288 | 8 | ~250K+ |
| 48GB (A6000) | 16384 | 6 | ~300K+ |

### Advanced Optimization Parameters
```bash
# Maximum performance configuration
--env.sim.nconmax 100      # Increase max contacts (default: 50)
--env.sim.njmax 600         # Increase max Jacobian rows (default: 300)
--agent.num_steps_per_env 32  # Longer rollouts (default: 24)
--agent.algorithm.num_learning_epochs 5  # More PPO epochs (default: 5)
--agent.algorithm.gamma 0.99  # Discount factor
--agent.algorithm.desired_kl 0.01  # KL divergence target
```

### Training Scripts - Ready to Use!
```bash
# Single GPU optimized training (run this!)
./train_asimov_a6000.sh         # Flat terrain, 8192 envs
./train_asimov_a6000.sh rough   # Rough terrain, 8192 envs
./train_asimov_a6000.sh flat 12288  # Flat, 12288 envs

# Dual GPU parallel training (maximize both GPUs!)
./train_asimov_dual.sh          # Trains flat & rough simultaneously
```

### Baseline Configurations (for comparison)
```bash
# Baseline Asimov (ideal actuators) - 8192 envs
uv run train Mjlab-Velocity-Flat-Asimov --env.scene.num_envs 8192

# Baseline Asimov (ideal actuators) - 12288 envs
uv run train Mjlab-Velocity-Flat-Asimov --env.scene.num_envs 12288
```

## Quick Summary - What's Special Here

### 🚀 Your Setup is POWERFUL!
- **Learned Actuator Dynamics**: Real motor behavior from CAN bus data (0.48° accuracy!)
- **Dual RTX A6000 GPUs**: 96GB total VRAM - run 2-3× more environments than typical setups
- **Optimized for Asimov**: Configured specifically for your robot, not generic configs
- **Ready to Train**: Just run `./train_asimov_a6000.sh` to start!

### The `Mjlab-Velocity-Flat-Asimov-Learned` Config
YES, this runs WITH your trained actuator network! It:
- Uses realistic motor dynamics (request-response latency from CAN data)
- Implements 50Hz→200Hz step-wise command holding
- Simulates actual motor delays and tracking errors
- Provides better sim-to-real transfer than ideal actuators

## 🎯 Tanh Actor for Bounded Actions

### Why Tanh?
- Currently actions are unbounded (linear output layer)
- Actions are only clipped AFTER network output
- Tanh bounds actions smoothly to [-1, 1] DURING network forward pass
- Better gradient flow and training stability

### Implementation
- **File**: `src/mjlab/rl/actors/tanh_actor.py`
- Adds tanh activation to final layer of actor network
- Drop-in replacement for standard actor

### How to Use
```python
# In your training config or script
from mjlab.rl.actors.tanh_actor import TanhActor

# The actor will automatically use tanh on output
# Actions will be in range [-1, 1] before scaling
```

## 🤖 Sim2Real Transfer - CRITICAL CHECKLIST

### Observation Configuration (45 elements)
**FIXED**: Training now uses 45 observations (removed `base_lin_vel`) to match firmware deployment.

**Observation order** (must match firmware exactly):
```
1. base_ang_vel (3)        - Angular velocity × 0.25
2. projected_gravity (3)   - Gravity in body frame
3. command (3)             - [vx, vy, wz] velocity commands
4. joint_pos (12)          - Relative to default positions
5. joint_vel (12)          - Joint velocities × 0.05
6. actions (12)            - Previous actions (raw NN output)
Total: 45 elements
```

### Action Configuration
**Per-joint action scales** (from `asimov_constants.py`):
- hip_pitch: 1.026650 (formula: 0.3 × effort_limit / stiffness)
- hip_roll/knee: 0.657490
- hip_yaw: 1.026650
- ankle_pitch/roll: 0.822332

**Action formula**: `target_pos = raw_action × scale + default_position`

**Default joint positions** (KNEES_BENT_KEYFRAME):
```python
# Left leg
left_hip_pitch:   0.2,    left_knee: -0.4,    left_ankle_pitch: -0.25
left_hip_roll:    0.0,    left_ankle_roll: 0.0
left_hip_yaw:     0.0

# Right leg (note sign differences for canted hips!)
right_hip_pitch: -0.2,    right_knee:  0.4,    right_ankle_pitch:  0.25
right_hip_roll:   0.0,    right_ankle_roll: 0.0
right_hip_yaw:    0.0
```

### Firmware Deployment Checklist
Before deploying to real robot:

- [ ] **Verify observation size**: 45 elements (NOT 48)
- [ ] **Verify observation ordering** matches firmware exactly
- [ ] **Verify scaling factors**: ang_vel×0.25, joint_vel×0.05, joint_pos×1.0
- [ ] **Verify per-joint action scales** (NOT a single global scale)
- [ ] **Verify default joint positions** match KNEES_BENT_KEYFRAME exactly
- [ ] **Verify joint ordering** matches MJCF model (12 joints, left then right)
- [ ] **Verify IMU axis transforms**: x=-y_imu, y=-x_imu, z=z_imu
- [ ] **Verify gravity direction** when robot is level
- [ ] **Export ONNX with metadata** for verification
- [ ] **Test with zero commands** first (standing pose)
- [ ] **Test with small forward velocity** (0.1 m/s)

### Critical Sim2Real Issues
1. **Observation order**: Firmware puts `command` at index 6-8, matches new training config
2. **Action scales**: Must be per-joint, not global
3. **Default positions**: Wrong offsets will cause incorrect poses
4. **Gravity**: Firmware uses IMU gravity directly (body frame)

## Key Fixes Applied
1. Contact sensor: Removed `secondary=ContactMatch(mode="body", pattern="terrain")`
   - Changed to `secondary=None` to avoid terrain body reference error
2. Import paths: Fixed all relative imports to use absolute imports with `mjlab.`
3. Entity data attributes: Changed `joint_q`→`joint_pos`, `joint_qd`→`joint_vel`
4. **SIM2REAL**: Removed `base_lin_vel` observation (48→45 elements) to match firmware