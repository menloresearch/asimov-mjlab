from dataclasses import dataclass

from mjlab.envs.mdp.actions import joint_actions
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
  actuator_names: tuple[str, ...]
  """Tuple of actuator names or regex expressions that the action will be mapped to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
  offset: float | dict[str, float] = 0.0
  """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
  preserve_order: bool = False
  """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@dataclass(kw_only=True)
class JointPositionActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointPositionAction
  use_default_offset: bool = True


#
# Ankle AB (tendon) action mapping configuration
#


@dataclass(kw_only=True)
class AnklePrToTendonActionCfg(ActionTermCfg):
  """Map ankle pitch/roll inputs to A/B tendon position targets.

  The action term consumes 4 inputs ordered as:
  [left_pitch, left_roll, right_pitch, right_roll]

  It outputs 4 position targets, applied to actuators that control the
  A/B tendons on left and right ankles in this order:
  [left_A, left_B, right_A, right_B].

  The mapping is a linearized model with geometry parameters L and d:
    left_A  = -L * left_pitch  - d * left_roll
    left_B  = -L * left_pitch  + d * left_roll
    right_A = +L * right_pitch - d * right_roll
    right_B = +L * right_pitch + d * right_roll

  Notes:
  - Actuator names must exist in the asset (tendon or joint actuators).
  - Scale/offset can be scalar (applied to all) or a dict keyed by the
    joint input names to set per-input factors.
  """

  # Required: names in the asset
  left_pitch_joint: str
  left_roll_joint: str
  right_pitch_joint: str
  right_roll_joint: str

  left_tendon_A: str
  left_tendon_B: str
  right_tendon_A: str
  right_tendon_B: str

  # Optional scaling/offset for the 4 PR inputs
  scale: float | dict[str, float] = 1.0
  offset: float | dict[str, float] = 0.0
  use_default_offset: bool = False

  # Geometry parameters for the PR->AB linear mapping
  L: float = 1.0
  d: float = 1.0

  # Implementation type
  from mjlab.envs.mdp.actions.ankle_ab_action import AnklePrToTendonAction

  class_type: type[ActionTerm] = AnklePrToTendonAction
