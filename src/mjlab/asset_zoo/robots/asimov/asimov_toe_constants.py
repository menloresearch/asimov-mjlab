"""Asimov bipedal robot with toe joints constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

ASIMOV_TOE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "asimov" / "xmls" / "asimov_toe.xml"
)
assert ASIMOV_TOE_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ASIMOV_TOE_XML.parent.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ASIMOV_TOE_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs - Using same motors as Booster T1 / Unitree G1 legs
# Based on 7520 series actuators used in leg joints

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

# Ankle actuators - using similar specs as 5020 actuators but doubled for parallel linkage
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)

# Lower natural frequency for lighter robot (50% mass of T1)
# This reduces stiffness and makes control smoother
NATURAL_FREQ = 8 * 2.0 * 3.1415926535  # 8Hz (was 10Hz for G1)
DAMPING_RATIO = 1.8  # Slightly lower for lighter robot

STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2

DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ

# Asimov joint actuator configuration
# Hip pitch and hip yaw use 7520_14 actuators
ASIMOV_ACTUATOR_HIP_PITCH_YAW = ActuatorCfg(
  joint_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
)

# Hip roll and knee use 7520_22 actuators (more powerful)
ASIMOV_ACTUATOR_HIP_ROLL_KNEE = ActuatorCfg(
  joint_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
)

# Ankle joints - using doubled 5020 actuators for parallel linkage
ASIMOV_ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
)

# Toe joints - passive spring with low control authority
ASIMOV_TOE_ACTUATOR = ActuatorCfg(
  joint_names_expr=("left_toe_joint", "right_toe_joint"),
  effort_limit=5.0,  # From URDF
  armature=0.0001,  # Minimal inertia
  stiffness=50.0,   # From URDF spring_stiffness
  damping=0.8,      # From URDF damping
)

##
# Keyframe config.
##

# Default standing pose for Asimov (legs straight)
STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.71),  # Adjusted for toe geometry at ground level
  joint_pos={
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,   # Left: 0 = straight
    "right_knee_joint": 0.0,  # Right: 0 = straight (both at neutral)
    ".*_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    # Toe joints - try reference position (0.0) first
    "left_toe_joint": 0.0,   # At reference position
    "right_toe_joint": 0.0,   # At reference position
  },
  joint_vel={".*": 0.0},
)

# Knees bent pose for stability
# NOTE: Coordinate system corrected to match real hardware (Alex's update)
# Knees:
# - Left knee: axis=(0, 1, 0), range=[0, 1.5] → positive = extend back
# - Right knee: axis=(0, -1, 0), range=[-1.5, 0] → negative = extend back
# Ankle pitch:
# - Left ankle: axis=(0, 1, 0) → positive = pitch up
# - Right ankle: axis=(0, -1, 0) → negative = pitch up
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.73),  # Adjusted for toe geometry at ground level
  joint_pos={
    # Based on proven booster_gym Asimov_unitree.yaml config
    "left_hip_pitch_joint": 0.2,      # More upright than G1
    "right_hip_pitch_joint": -0.2,    # Opposite due to canted axis
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,           # Left: positive = extend backwards (corrected)
    "right_knee_joint": -0.4,         # Right: negative = extend backwards (corrected)
    "left_ankle_pitch_joint": -0.25,  # From v1.yaml config
    "right_ankle_pitch_joint": 0.25,  # Opposite axis
    ".*_ankle_roll_joint": 0.0,
    # Toe joints - try reference position (0.0) first
    "left_toe_joint": 0.0,            # At reference position
    "right_toe_joint": 0.0,            # At reference position
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Enable all collisions including self collisions
# Feet get condim=3, other parts get condim=1
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_ankle_roll_link_collision$": 3,
    ".*_collision": 1
  },
  priority={r"^(left|right)_ankle_roll_link_collision$": 1},
  friction={r"^(left|right)_ankle_roll_link_collision$": (0.8,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_ankle_roll_link_collision$": 3,
    ".*_collision": 1
  },
  priority={r"^(left|right)_ankle_roll_link_collision$": 1},
  friction={r"^(left|right)_ankle_roll_link_collision$": (0.8,)},
)

# Only feet collision enabled (recommended for initial training)
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_ankle_roll_link_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

ASIMOV_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ASIMOV_ACTUATOR_HIP_PITCH_YAW,
    ASIMOV_ACTUATOR_HIP_ROLL_KNEE,
    ASIMOV_ACTUATOR_ANKLE,
    ASIMOV_TOE_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_asimov_robot_cfg() -> EntityCfg:
  """Get a fresh Asimov robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ASIMOV_ARTICULATION,
  )


# Compute action scales for each joint
# Using 0.3 multiplier (vs 0.25 for G1) due to lighter mass and different kinematics
ASIMOV_ACTION_SCALE: dict[str, float] = {}
for a in ASIMOV_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}
  for n in names:
    if n in e and n in s and s[n]:
      # 0.3 multiplier (vs G1's 0.25) for more responsive control on lighter robot
      ASIMOV_ACTION_SCALE[n] = 0.3 * e[n] / s[n]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_asimov_robot_cfg())

  viewer.launch(robot.spec.compile())
