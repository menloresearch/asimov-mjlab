"""Asimov bipedal robot velocity tracking with learned actuator dynamics.

This configuration uses the neural network model trained on CAN bus data
to simulate realistic motor dynamics including request-response latency.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots.asimov.asimov_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg,
  get_asimov_robot_cfg_with_learned_actuator,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval


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

  # Use robot cfg with learned actuator network
  robot_cfg = get_asimov_robot_cfg_with_learned_actuator(
    network_file="actuator_torque_checkpoints/actuator_network.pt"
  )

  cfg = create_velocity_env_cfg(
    robot_cfg=robot_cfg,
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

  # SIM2REAL FIX: Remove base_lin_vel observation to match firmware (45 obs instead of 48)
  # Firmware observation structure: [base_ang_vel(3), projected_gravity(3), command(3),
  #                                   joint_pos(12), joint_vel(12), actions(12)] = 45
  assert cfg.observations is not None
  policy_obs = cfg.observations["policy"]
  critic_obs = cfg.observations["critic"]

  # Remove base_lin_vel from both policy and critic
  policy_obs.terms.pop("base_lin_vel", None)
  critic_obs.terms.pop("base_lin_vel", None)

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