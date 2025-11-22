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


#
# Learned actuator action configuration
#


@dataclass(kw_only=True)
class LearnedActuatorActionCfg(JointActionCfg):
  """Configuration for learned actuator dynamics action.

  This action term uses a neural network trained on CAN bus data to simulate
  realistic motor dynamics including:
  - Request-response protocol latency
  - Motor processing delays
  - Temperature-dependent performance
  - Tracking errors and systematic biases

  The model was trained on real hardware data at 200Hz with 50Hz command updates.
  """

  from mjlab.envs.mdp.actions.learned_actuator_action import LearnedActuatorAction

  class_type: type[ActionTerm] = LearnedActuatorAction

  model_path: str = "actuator_checkpoints/best_model.pth"
  """Path to the trained actuator model checkpoint."""

  window_size: int = 8
  """Number of historical timesteps for temporal context (8 * 5ms = 40ms window)."""

  control_decimation: int = 4
  """Number of simulation steps per policy update (4 * 5ms = 20ms = 50Hz)."""

  use_position_target: bool = True
  """If True, write position targets. If False, directly set joint states."""

  use_default_offset: bool = True
  """Whether to use default joint positions as offset."""
