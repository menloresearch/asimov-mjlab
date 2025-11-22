"""Asimov bipedal robot velocity tracking with learned actuator dynamics.

This configuration uses the neural network model trained on CAN bus data
to simulate realistic motor dynamics including request-response latency.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots.asimov.asimov_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval

# Import the learned actuator configuration
from mjlab.envs.mdp.actions import LearnedActuatorActionCfg


@retval
def ASIMOV_ROUGH_ENV_CFG_LEARNED() -> ManagerBasedRlEnvCfg:
  """Create Asimov rough terrain velocity tracking with learned actuators."""
  # Asimov feet sites at ankle roll joints
  site_names = ("left_ankle_roll_joint_site", "right_ankle_roll_joint_site")
  geom_names = (
    "left_ankle_roll_link_collision",
    "right_ankle_roll_link_collision",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=None,  # Fixed: removed terrain body reference that caused compilation error
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = create_velocity_env_cfg(
    robot_cfg=get_asimov_robot_cfg(),
    action_scale=deepcopy(ASIMOV_ACTION_SCALE),
    viewer_body_name="pelvis_link",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=self_collision_cfg,
    foot_friction_geom_names=geom_names,
    posture_std_standing={".*": 0.05},
    posture_std_walking={
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.25,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.5,
      r".*ankle_pitch.*": 0.2,
      r".*ankle_roll.*": 0.12,
    },
    posture_std_running={
      r".*hip_pitch.*": 0.8,
      r".*hip_roll.*": 0.35,
      r".*hip_yaw.*": 0.3,
      r".*knee.*": 0.8,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.15,
    },
    # Asimov-specific reward weights
    body_ang_vel_weight=-0.08,
    angular_momentum_weight=-0.03,
    self_collision_weight=-1.0,
    air_time_weight=0.5,
  )

  # Configure scene for 4096 environments
  cfg.scene.num_envs = 4096

  # MODIFICATION: Replace standard JointPositionAction with LearnedActuatorAction
  cfg.actions = {
    "joint_pos": LearnedActuatorActionCfg(
      asset_name="robot",
      actuator_names=(".*",),  # All actuators
      scale=ASIMOV_ACTION_SCALE,
      use_default_offset=True,
      # Learned actuator specific parameters
      model_path="actuator_checkpoints/best_model.pth",
      window_size=8,  # 40ms history window
      control_decimation=4,  # 50Hz policy, 200Hz simulation
      use_position_target=False,  # Directly set joint states for more accurate dynamics
    )
  }

  # Configure visualization
  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8  # Asimov is shorter than G1

  # Specific tuning for velocity command ranges (narrower for Asimov's lighter build)
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)  # m/s (reduced from G1's (-1.5, 1.5))
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)  # m/s (reduced from G1's (-0.8, 0.8))
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)  # rad/s (reduced from G1's (-1.0, 1.0))

  # OPTIONAL: Adjust reward weights if needed for learned dynamics
  # The learned actuator may have different characteristics that benefit from tuning
  # Uncomment and adjust if training is unstable
  # cfg.rewards.action_rate.weight = -0.005  # Might need less penalty with realistic delays
  # cfg.rewards.joint_acc.weight = -1.0e-07  # May need adjustment for learned dynamics

  return cfg


@retval
def ASIMOV_FLAT_ENV_CFG_LEARNED() -> ManagerBasedRlEnvCfg:
  """Create Asimov flat terrain velocity tracking with learned actuators."""
  cfg = deepcopy(ASIMOV_ROUGH_ENV_CFG_LEARNED)
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove terrain curriculum for flat terrain (matching working config pattern)
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  return cfg