# Asimov Sim2Real Transfer Guide

## ✅ Configuration Status

**VERIFIED**: All Asimov training configurations now use **45 observations** matching firmware deployment.

Run `uv run python verify_obs_size.py` to verify at any time.

## Observation Structure (45 elements)

```
Index | Observation          | Size | Scaling | Source
------|---------------------|------|---------|------------------
0-2   | base_ang_vel        | 3    | ×0.25   | IMU angular velocity
3-5   | projected_gravity   | 3    | ×1.0    | Gravity in body frame
6-8   | command             | 3    | ×1.0    | [vx, vy, wz] commands
9-20  | joint_pos           | 12   | ×1.0    | Relative to default
21-32 | joint_vel           | 12   | ×0.05   | Joint velocities
33-44 | actions             | 12   | ×1.0    | Previous raw actions
------|---------------------|------|---------|------------------
TOTAL |                     | 45   |         |
```

## Action Configuration

### Per-Joint Action Scales

From `asimov_constants.py`, formula: `scale = 0.3 × effort_limit / stiffness`

| Joint Type | Scale | Effort Limit | Stiffness |
|------------|-------|--------------|-----------|
| hip_pitch  | 1.026650 | 88.0 | 25.768 |
| hip_yaw    | 1.026650 | 88.0 | 25.768 |
| hip_roll   | 0.657490 | 139.0 | 63.434 |
| knee       | 0.657490 | 139.0 | 63.434 |
| ankle_pitch | 0.822332 | 50.0 | 18.254 |
| ankle_roll | 0.822332 | 50.0 | 18.254 |

### Action Processing Formula

```c
target_position = raw_action × action_scale + default_position
```

### Default Joint Positions (KNEES_BENT_KEYFRAME)

```python
# Left leg
left_hip_pitch:   0.2      # More upright
left_hip_roll:    0.0
left_hip_yaw:     0.0
left_knee:       -0.4      # Negative = extend backwards
left_ankle_pitch: -0.25
left_ankle_roll:  0.0

# Right leg (note sign differences due to canted hip design!)
right_hip_pitch:  -0.2     # Opposite due to canted axis
right_hip_roll:   0.0
right_hip_yaw:    0.0
right_knee:       0.4      # Positive = extend backwards
right_ankle_pitch: 0.25    # Opposite axis
right_ankle_roll: 0.0
```

## Joint Ordering (MJCF Model)

```
Motor ID | Joint Name
---------|------------------------
0        | left_hip_pitch_joint
1        | left_hip_roll_joint
2        | left_hip_yaw_joint
3        | left_knee_joint
4        | left_ankle_pitch_joint
5        | left_ankle_roll_joint
6        | right_hip_pitch_joint
7        | right_hip_roll_joint
8        | right_hip_yaw_joint
9        | right_knee_joint
10       | right_ankle_pitch_joint
11       | right_ankle_roll_joint
```

## Firmware Checklist

Before deploying to real robot:

### ✅ Observation Configuration
- [ ] **Size**: 45 elements (NOT 48 - base_lin_vel removed)
- [ ] **Ordering**: [ang_vel, gravity, command, pos, vel, actions]
- [ ] **Scaling**: ang_vel×0.25, joint_vel×0.05, joint_pos×1.0
- [ ] **IMU axis transform**: x=-y_imu, y=-x_imu, z=z_imu
- [ ] **Gravity direction**: Verify it points down when level

### ✅ Action Configuration
- [ ] **Per-joint scales**: Use table above (NOT a single global scale)
- [ ] **Default positions**: Match KNEES_BENT_KEYFRAME exactly
- [ ] **Action formula**: `target = raw × scale + default`
- [ ] **Joint ordering**: Matches MJCF model (0-11)

### ✅ ONNX Export
- [ ] **Export with metadata**: Includes action_scale, default_joint_pos, joint_names
- [ ] **Verify input size**: 45 observations
- [ ] **Verify output size**: 12 actions
- [ ] **Test inference**: Forward pass with dummy inputs

### ✅ Safety Testing
- [ ] **Zero commands first**: Robot should hold standing pose
- [ ] **Small forward velocity**: Test with 0.1 m/s
- [ ] **Verify motion direction**: Forward command moves robot forward
- [ ] **Emergency stop ready**: Test E-stop before full testing

## Training Commands

### Start Fresh Training (REQUIRED after config changes)

```bash
# Train with learned actuator dynamics (RECOMMENDED)
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Asimov-Learned \
    --env.scene.num_envs 4096 \
    --agent.max_iterations 10000

# Train on rough terrain
MUJOCO_GL=egl uv run train Mjlab-Velocity-Rough-Asimov-Learned \
    --env.scene.num_envs 4096 \
    --agent.max_iterations 10000

# Train baseline (ideal actuators) for comparison
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Asimov \
    --env.scene.num_envs 4096 \
    --agent.max_iterations 10000
```

## Common Sim2Real Issues

### Issue 1: Robot Tries to Reach Wrong Poses
**Cause**: Default joint positions don't match training
**Fix**: Verify `default_angle` array in firmware matches KNEES_BENT_KEYFRAME

### Issue 2: Actions Too Large/Small
**Cause**: Using wrong action scales (global vs per-joint)
**Fix**: Use per-joint scales from table above

### Issue 3: Robot Falls Immediately
**Cause**: Observation ordering mismatch
**Fix**: Verify firmware builds observations in exact order above

### Issue 4: Unstable Behavior
**Cause**: IMU axis transformation incorrect
**Fix**: Verify axis swapping: x=-y_imu, y=-x_imu, z=z_imu

### Issue 5: Wrong Movement Direction
**Cause**: Command ordering or sign mismatch
**Fix**: Verify commands are [vx_forward, vy_left, wz_ccw]

## ONNX Export Example

```python
from mjlab.tasks.velocity.rl.exporter import export_policy_as_onnx

# Export with metadata
export_policy_as_onnx(
    policy=runner.get_inference_policy(),
    path="path/to/output",
    filename="asimov_policy.onnx"
)

# Metadata includes:
# - joint_names: Joint ordering
# - action_scale: Per-joint scales
# - default_joint_pos: Default positions
# - joint_stiffness: PD controller Kp
# - joint_damping: PD controller Kd
```

## Verification Commands

```bash
# Verify observation sizes
uv run python verify_obs_size.py

# Test policy in simulation
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
    --checkpoint-file logs/rsl_rl/asimov_velocity/<run>/model_<iter>.pt

# Record video for analysis
MUJOCO_GL=egl uv run play Mjlab-Velocity-Flat-Asimov-Learned \
    --checkpoint-file logs/rsl_rl/asimov_velocity/<run>/model_<iter>.pt \
    --video True \
    --num-envs 1
```

## Key Changes Made

1. **Removed `base_lin_vel` observation** (48→45 elements)
   - Modified: `src/mjlab/tasks/velocity/config/asimov/env_cfgs.py`
   - Modified: `src/mjlab/tasks/velocity/config/asimov/env_cfgs_learned.py`

2. **Added verification script**: `verify_obs_size.py`

3. **Updated documentation**: `CLAUDE.md` with sim2real checklist

## Next Steps

1. ✅ **Verify configs**: `uv run python verify_obs_size.py`
2. **Train new policy**: Use commands above (existing policies have wrong obs size)
3. **Export ONNX**: With metadata for deployment
4. **Test in sim**: Verify behavior before deployment
5. **Deploy to robot**: Follow safety checklist above

## References

- Training observation config: `src/mjlab/tasks/velocity/velocity_env_cfg.py`
- Asimov constants: `src/mjlab/asset_zoo/robots/asimov/asimov_constants.py`
- Action processing: `src/mjlab/envs/mdp/actions/joint_actions.py`
- Gravity projection: `src/mjlab/entity/data.py:500-502`
