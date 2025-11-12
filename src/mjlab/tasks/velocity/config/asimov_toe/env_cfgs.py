"""Asimov bipedal robot with toe joints velocity tracking environment configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots.asimov.asimov_toe_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval


@retval
def ASIMOV_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Asimov rough terrain velocity tracking configuration."""
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

  robot_cfg = get_asimov_robot_cfg()

  # Create action scale WITHOUT toe joints (keep them passive)
  action_scale_no_toes = {k: v for k, v in ASIMOV_ACTION_SCALE.items()
                          if 'toe' not in k}

  cfg = create_velocity_env_cfg(
    robot_cfg=robot_cfg,
    action_scale=action_scale_no_toes,  # Only control 12 DoF, toes are passive
    viewer_body_name="pelvis_link",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=self_collision_cfg,
    foot_friction_geom_names=geom_names,
    posture_std_standing={".*": 0.05},
    posture_std_walking={
      # Larger variance for canted hip pitch (allows coupled roll/pitch motion)
      r".*hip_pitch.*": 0.5,
      # Hip roll: reduced for ±45° range (was wider before Alex's corrections)
      r".*hip_roll.*": 0.12,
      # Hip yaw: reduced for ±45° range (was wider before Alex's corrections)
      r".*hip_yaw.*": 0.1,
      # Large knee variance (coordinate system corrected to match hardware)
      r".*knee.*": 0.5,
      # Lower ankle variance due to limited ROM (~±20° roll, asymmetric pitch)
      r".*ankle_pitch.*": 0.2,
      r".*ankle_roll.*": 0.12,
      # Toe joints - passive, low variance
      r".*toe.*": 0.3,
    },
    posture_std_running={
      # Even larger variance for dynamic motion with canted hips
      r".*hip_pitch.*": 0.8,
      # Hip roll: reduced for ±45° range (was wider before Alex's corrections)
      r".*hip_roll.*": 0.18,
      # Hip yaw: reduced for ±45° range (was wider before Alex's corrections)
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.8,
      # Keep ankles constrained even when running
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.15,
      # Toe joints - allow more motion during running
      r".*toe.*": 0.4,
    },
    # Increase body angular velocity penalty (narrow stance = less stable)
    body_ang_vel_weight=-0.08,
    angular_momentum_weight=-0.03,
    self_collision_weight=-1.0,
    # Enable air time reward for lighter robot (better for jumping)
    air_time_weight=0.5,
  )

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8  # Asimov is shorter than G1

  # More conservative velocity commands due to:
  # 1. Narrow stance (11.3 cm hip width)
  # 2. Canted hip pitch (less stable)
  # 3. Limited ankle ROM
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)  # Reduced from (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)  # Reduced from (-1.0, 1.0) - narrow stance
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)  # Slightly reduced from (-0.5, 0.5)

  return cfg


@retval
def ASIMOV_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Asimov flat terrain velocity tracking configuration."""
  # Start with rough terrain config.
  cfg = deepcopy(ASIMOV_ROUGH_ENV_CFG)

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  return cfg
