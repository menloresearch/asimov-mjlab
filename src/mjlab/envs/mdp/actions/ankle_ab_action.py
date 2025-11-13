from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.actions.actions_config import AnklePrToTendonActionCfg


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

