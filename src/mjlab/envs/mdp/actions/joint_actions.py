from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.actions import actions_config


class JointAction(ActionTerm):
  """Base class for joint actions."""

  _asset: Entity

  def __init__(self, cfg: actions_config.JointActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    actuator_ids, self._actuator_names = self._asset.find_actuators(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    joint_ids, _ = self._asset.find_joints(
      self._actuator_names, preserve_order=cfg.preserve_order
    )

    self._actuator_ids = torch.tensor(
      actuator_ids, device=self.device, dtype=torch.long
    )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    self._num_joints = len(self._actuator_ids)
    self._action_dim = len(self._actuator_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._actuator_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}."
        " Supported types are float and dict."
      )

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._actuator_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}."
        " Supported types are float and dict."
      )

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    return self._action_dim

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
  def __init__(self, cfg: actions_config.JointPositionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

  def apply_actions(self):
    self._asset.write_joint_position_target_to_sim(
      self._processed_actions, self._actuator_ids
    )


class VariableImpedanceJointPositionAction(JointAction):
  """Joint position action with per-group variable stiffness and damping.

  Action space is split into three parts:
  - Position targets: [num_joints]
  - Stiffness scales: [num_groups]
  - Damping scales: [num_groups]

  Where num_groups is the number of actuator groups (e.g., hip_pitch_yaw, hip_roll_knee).
  """

  def __init__(
    self, cfg: actions_config.VariableImpedanceJointPositionActionCfg, env: ManagerBasedEnv
  ):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    # Store base stiffness and damping values from the model.
    self._base_stiffness = self._asset.data.default_joint_stiffness[
      :, self._actuator_ids
    ].clone()
    self._base_damping = self._asset.data.default_joint_damping[
      :, self._actuator_ids
    ].clone()

    # Build group mapping: which actuators belong to which group.
    # cfg.stiffness_groups is a dict: {group_name: actuator_regex_list}
    self._num_groups = len(cfg.stiffness_groups)
    self._group_to_actuator_indices = {}

    from mjlab.third_party.isaaclab.isaaclab.utils.string import (
      resolve_matching_names,
    )

    for group_idx, (group_name, actuator_patterns) in enumerate(cfg.stiffness_groups.items()):
      # Find which actuators match this group's patterns.
      matched_indices = []
      for pattern in actuator_patterns:
        indices = resolve_matching_names(pattern, self._actuator_names)[0]
        matched_indices.extend(indices)

      self._group_to_actuator_indices[group_idx] = torch.tensor(
        matched_indices, device=self.device, dtype=torch.long
      )

    # Update action dimension to include position targets + kp scales + kd scales.
    self._position_action_dim = self._num_joints
    self._action_dim = self._position_action_dim + 2 * self._num_groups

    # Recreate raw action buffers with new dimension (base class created them with wrong size).
    # Note: _raw_actions only stores position targets (not gains).
    # The base class already created these, so we don't need to recreate them.
    # They should stay at _num_joints size for position targets only.

    # Buffers for kp/kd scales.
    self._kp_scales = torch.ones(self.num_envs, self._num_groups, device=self.device)
    self._kd_scales = torch.ones(self.num_envs, self._num_groups, device=self.device)

    # Clamp ranges for gains.
    self._kp_scale_range = cfg.stiffness_scale_range
    self._kd_scale_range = cfg.damping_scale_range

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def kp_scales(self) -> torch.Tensor:
    """Current stiffness scales for each group. Shape: (num_envs, num_groups)"""
    return self._kp_scales

  @property
  def kd_scales(self) -> torch.Tensor:
    """Current damping scales for each group. Shape: (num_envs, num_groups)"""
    return self._kd_scales

  def process_actions(self, actions: torch.Tensor):
    # Split actions into: [positions, kp_scales, kd_scales].
    positions = actions[:, : self._position_action_dim]
    kp_scales = actions[:, self._position_action_dim : self._position_action_dim + self._num_groups]
    kd_scales = actions[:, self._position_action_dim + self._num_groups :]

    # Store and clamp gain scales.
    self._kp_scales[:] = torch.clamp(kp_scales, *self._kp_scale_range)
    self._kd_scales[:] = torch.clamp(kd_scales, *self._kd_scale_range)

    # Process position actions as usual.
    self._raw_actions[:] = positions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self):
    # Write position targets.
    self._asset.write_joint_position_target_to_sim(
      self._processed_actions, self._actuator_ids
    )

    # Update actuator gains in the model.
    # MuJoCo stores gains in model.actuator_gainprm[:, actuator_id, 0] for kp
    # and model.actuator_biasprm[:, actuator_id, 2] for -kd.
    for group_idx in range(self._num_groups):
      actuator_indices = self._group_to_actuator_indices[group_idx]

      # Get kp/kd scales for this group.
      kp_scale = self._kp_scales[:, group_idx : group_idx + 1]  # (num_envs, 1)
      kd_scale = self._kd_scales[:, group_idx : group_idx + 1]  # (num_envs, 1)

      # Compute new gains for all actuators in this group.
      # Clone to avoid expanded tensor warnings when writing back.
      base_kp_group = self._base_stiffness[:, actuator_indices].clone()  # (num_envs, num_actuators_in_group)
      base_kd_group = self._base_damping[:, actuator_indices].clone()    # (num_envs, num_actuators_in_group)

      new_kp_group = base_kp_group * kp_scale  # Broadcasting: (num_envs, N) * (num_envs, 1)
      new_kd_group = base_kd_group * kd_scale

      # Get global actuator IDs for this group.
      ctrl_ids_group = self._actuator_ids[actuator_indices]  # (num_actuators_in_group,)

      # Batched write to model tensors.
      self._asset.data.model.actuator_gainprm[:, ctrl_ids_group, 0] = new_kp_group
      self._asset.data.model.actuator_biasprm[:, ctrl_ids_group, 2] = -new_kd_group

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # Reset gain scales to 1.0.
    self._kp_scales[env_ids] = 1.0
    self._kd_scales[env_ids] = 1.0
