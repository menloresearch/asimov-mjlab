"""Asimov bipedal robot velocity tracking with learned actuator dynamics.

This configuration uses the neural network model trained on CAN bus data
to simulate realistic motor dynamics including request-response latency.
"""

from mjlab.asset_zoo.robots.asimov.asimov_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg_with_learned_actuator,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def ASIMOV_ROUGH_ENV_CFG_LEARNED(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Asimov rough terrain velocity tracking with learned actuators."""
  cfg = make_velocity_env_cfg()

  # Use robot cfg with learned actuator network
  cfg.scene.entities = {
    "robot": get_asimov_robot_cfg_with_learned_actuator(
      network_file="actuator_torque_checkpoints/actuator_network.pt"
    )
  }

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
    secondary=ContactMatch(mode="body", pattern="terrain"),
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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ASIMOV_ACTION_SCALE

  cfg.viewer.body_name = "pelvis_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8  # Match Asimov base height

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.2,
    r".*ankle_roll.*": 0.12,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.8,
    r".*hip_roll.*": 0.35,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.8,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Asimov-specific reward weights
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["air_time"].weight = 0.5

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # SIM2REAL FIX: Remove base_lin_vel observation to match firmware (45 obs instead of 48)
  # Firmware observation structure: [base_ang_vel(3), projected_gravity(3), command(3),
  #                                   joint_pos(12), joint_vel(12), actions(12)] = 45
  assert cfg.observations is not None
  policy_obs = cfg.observations["policy"]
  critic_obs = cfg.observations["critic"]

  # Remove base_lin_vel from both policy and critic
  policy_obs.terms.pop("base_lin_vel", None)
  critic_obs.terms.pop("base_lin_vel", None)

  # Velocity command ranges tuned for Asimov's lighter build and narrow stance
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)  # m/s - conservative for stability
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)  # m/s - limited by narrow 11.3cm hip width
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)  # rad/s - conservative for learning

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def ASIMOV_FLAT_ENV_CFG_LEARNED(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Asimov flat terrain velocity tracking with learned actuators."""
  cfg = ASIMOV_ROUGH_ENV_CFG_LEARNED(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  return cfg