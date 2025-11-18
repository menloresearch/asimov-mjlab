from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.actions.actions_config import (
    AnklePrToTendonActionCfg,
    VariableImpedanceAnklePrToTendonActionCfg,
  )


class AnklePrToTendonAction(ActionTerm):
  """Action term mapping ankle PR targets to AB tendon position targets.

  Input order: [left_pitch, left_roll, right_pitch, right_roll].
  Output controls: [left_A, left_B, right_A, right_B] tendon position targets.
  """

  def __init__(self, cfg: AnklePrToTendonActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    self._cfg = cfg
    self._asset = self._env.scene[self._cfg.asset_name]

    # Resolve joint names to indices (for default offsets and to keep names stable).
    joint_names = [
      self._cfg.left_pitch_joint,
      self._cfg.left_roll_joint,
      self._cfg.right_pitch_joint,
      self._cfg.right_roll_joint,
    ]
    joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    # Resolve tendon actuators (AB space) in order: left_A, left_B, right_A, right_B.
    actuator_names = [
      self._cfg.left_tendon_A,
      self._cfg.left_tendon_B,
      self._cfg.right_tendon_A,
      self._cfg.right_tendon_B,
    ]
    actuator_ids, _ = self._asset.find_actuators(actuator_names, preserve_order=True)
    self._actuator_ids = torch.tensor(
      actuator_ids, device=self.device, dtype=torch.long
    )

    # Buffers.
    self._num_vars = 4
    self._raw_actions = torch.zeros(self.num_envs, self._num_vars, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Scale.
    if isinstance(self._cfg.scale, (float, int)):
      self._scale = float(self._cfg.scale)
    elif isinstance(self._cfg.scale, dict):
      # Match using joint names (PR inputs) to preserve semantics.
      self._scale = torch.ones(self.num_envs, self._num_vars, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        self._cfg.scale, joint_names, preserve_order=True
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError("Unsupported scale type for AnklePrToTendonAction.")

    # Offset.
    if isinstance(self._cfg.offset, (float, int)):
      self._offset = float(self._cfg.offset)
    elif isinstance(self._cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        self._cfg.offset, joint_names, preserve_order=True
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError("Unsupported offset type for AnklePrToTendonAction.")

    # Use default joint positions as offset if requested.
    if self._cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    # Geometry parameters.
    self._L = float(self._cfg.L)
    self._d = float(self._cfg.d)

  # Properties.
  @property
  def action_dim(self) -> int:
    return self._num_vars

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  # Methods.
  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self) -> None:
    # Unpack PR inputs.
    theta_L = self._processed_actions[:, 0]
    phi_L = self._processed_actions[:, 1]
    theta_R = self._processed_actions[:, 2]
    phi_R = self._processed_actions[:, 3]

    L = self._L
    d = self._d

    # Linearized mapping to tendon position targets.
    # Left: yA = -L*theta - d*phi, yB = -L*theta + d*phi
    left_A = -L * theta_L - d * phi_L
    left_B = -L * theta_L + d * phi_L
    # Right: pitch sign flips due to opposite joint axis in XML; roll sign same.
    right_A = +L * theta_R - d * phi_R
    right_B = +L * theta_R + d * phi_R

    tendon_targets = torch.stack([left_A, left_B, right_A, right_B], dim=1)
    self._asset.data.write_ctrl(tendon_targets, self._actuator_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class VariableImpedanceAnklePrToTendonAction(ActionTerm):
  """Ankle PR->AB action with variable impedance control.

  Extends AnklePrToTendonAction to include per-group kp/kd scaling.
  Action space: [left_pitch, left_roll, right_pitch, right_roll, kp_scale, kd_scale]

  The stiffness_groups config determines which tendons share gain scales.
  Default: single group for all 4 tendons (left_A, left_B, right_A, right_B).
  """

  def __init__(
    self,
    cfg: AnklePrToTendonActionCfg | VariableImpedanceAnklePrToTendonActionCfg,
    env: ManagerBasedEnv,
  ):
    super().__init__(cfg=cfg, env=env)

    self._cfg = cfg
    self._asset = self._env.scene[self._cfg.asset_name]

    # Resolve joint names (PR inputs).
    joint_names = [
      self._cfg.left_pitch_joint,
      self._cfg.left_roll_joint,
      self._cfg.right_pitch_joint,
      self._cfg.right_roll_joint,
    ]
    joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    # Resolve tendon actuators (AB outputs).
    actuator_names = [
      self._cfg.left_tendon_A,
      self._cfg.left_tendon_B,
      self._cfg.right_tendon_A,
      self._cfg.right_tendon_B,
    ]
    actuator_ids, self._actuator_names = self._asset.find_actuators(
      actuator_names, preserve_order=True
    )
    self._actuator_ids = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)

    # Buffers for PR actions.
    self._num_pr_vars = 4
    self._raw_actions = torch.zeros(self.num_envs, self._num_pr_vars, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Scale and offset (same as base class).
    if isinstance(self._cfg.scale, (float, int)):
      self._scale = float(self._cfg.scale)
    elif isinstance(self._cfg.scale, dict):
      from mjlab.third_party.isaaclab.isaaclab.utils.string import (
        resolve_matching_names_values,
      )

      self._scale = torch.ones(self.num_envs, self._num_pr_vars, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        self._cfg.scale, joint_names, preserve_order=True
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError("Unsupported scale type.")

    if isinstance(self._cfg.offset, (float, int)):
      self._offset = float(self._cfg.offset)
    elif isinstance(self._cfg.offset, dict):
      from mjlab.third_party.isaaclab.isaaclab.utils.string import (
        resolve_matching_names_values,
      )

      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        self._cfg.offset, joint_names, preserve_order=True
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError("Unsupported offset type.")

    if self._cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    # Geometry parameters.
    self._L = float(self._cfg.L)
    self._d = float(self._cfg.d)

    # Variable impedance setup.
    if hasattr(cfg, "stiffness_groups"):
      # Store base gains for tendons.
      self._base_stiffness = self._asset.data.default_joint_stiffness[
        :, self._actuator_ids
      ].clone()
      self._base_damping = self._asset.data.default_joint_damping[:, self._actuator_ids].clone()

      # Build group mapping.
      from mjlab.third_party.isaaclab.isaaclab.utils.string import (
        resolve_matching_names,
      )

      self._num_groups = len(cfg.stiffness_groups)
      self._group_to_actuator_indices = {}

      for group_idx, (group_name, actuator_patterns) in enumerate(
        cfg.stiffness_groups.items()
      ):
        matched_indices = []
        for pattern in actuator_patterns:
          indices = resolve_matching_names(pattern, self._actuator_names)[0]
          matched_indices.extend(indices)

        self._group_to_actuator_indices[group_idx] = torch.tensor(
          matched_indices, device=self.device, dtype=torch.long
        )

      # Buffers for kp/kd scales.
      self._kp_scales = torch.ones(self.num_envs, self._num_groups, device=self.device)
      self._kd_scales = torch.ones(self.num_envs, self._num_groups, device=self.device)

      # Clamp ranges.
      self._kp_scale_range = cfg.stiffness_scale_range
      self._kd_scale_range = cfg.damping_scale_range

      # Update action dimension: PR inputs + kp scales + kd scales.
      self._action_dim = self._num_pr_vars + 2 * self._num_groups
    else:
      # No variable impedance.
      self._num_groups = 0
      self._action_dim = self._num_pr_vars

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def kp_scales(self) -> torch.Tensor | None:
    """Current stiffness scales for each group. Shape: (num_envs, num_groups). None if no variable impedance."""
    return self._kp_scales if self._num_groups > 0 else None

  @property
  def kd_scales(self) -> torch.Tensor | None:
    """Current damping scales for each group. Shape: (num_envs, num_groups). None if no variable impedance."""
    return self._kd_scales if self._num_groups > 0 else None

  def process_actions(self, actions: torch.Tensor) -> None:
    if self._num_groups > 0:
      # Variable impedance: split actions.
      pr_actions = actions[:, : self._num_pr_vars]
      kp_scales = actions[:, self._num_pr_vars : self._num_pr_vars + self._num_groups]
      kd_scales = actions[:, self._num_pr_vars + self._num_groups :]

      # Store and clamp gain scales.
      self._kp_scales[:] = torch.clamp(kp_scales, *self._kp_scale_range)
      self._kd_scales[:] = torch.clamp(kd_scales, *self._kd_scale_range)

      # Process PR actions.
      self._raw_actions[:] = pr_actions
      self._processed_actions = self._raw_actions * self._scale + self._offset
    else:
      # No variable impedance.
      self._raw_actions[:] = actions
      self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self) -> None:
    # Compute tendon targets from PR inputs (same as base class).
    theta_L = self._processed_actions[:, 0]
    phi_L = self._processed_actions[:, 1]
    theta_R = self._processed_actions[:, 2]
    phi_R = self._processed_actions[:, 3]

    L = self._L
    d = self._d

    left_A = -L * theta_L - d * phi_L
    left_B = -L * theta_L + d * phi_L
    right_A = +L * theta_R - d * phi_R
    right_B = +L * theta_R + d * phi_R

    tendon_targets = torch.stack([left_A, left_B, right_A, right_B], dim=1)
    self._asset.data.write_ctrl(tendon_targets, self._actuator_ids)

    # Update actuator gains if variable impedance is enabled.
    if self._num_groups > 0:
      for group_idx in range(self._num_groups):
        actuator_indices = self._group_to_actuator_indices[group_idx]

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
    self._raw_actions[env_ids] = 0.0
    if self._num_groups > 0:
      self._kp_scales[env_ids] = 1.0
      self._kd_scales[env_ids] = 1.0

