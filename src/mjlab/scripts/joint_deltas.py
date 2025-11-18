"""Script to analyze absolute joint positions from a trained policy - simplified version based on play.py."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from mjlab.viewer.base import EnvProtocol


@dataclass(frozen=True)
class AnalyzeConfig:
  checkpoint_file: str
  """Path to the trained checkpoint file (.pt)."""
  num_steps: int = 1000
  """Number of steps to collect data."""
  num_envs: int = 1
  """Number of parallel environments to run."""
  device: str | None = None
  """Device to run on (default: cuda:0 if available)."""
  output_dir: str = "./joint_position_analysis"
  """Directory to save histograms and data."""
  bins: int = 50
  """Number of bins for histograms."""
  viewer: Literal["none", "auto", "native", "viser"] = "none"
  """Viewer backend: none (headless analysis), auto (detect display), native (mujoco), viser (web-based)."""
  forward_speed: float = 0.8
  """Forward walking speed in m/s (default: 0.8)."""
  skip_initial_steps: int = 0
  """Number of initial steps to skip (to remove initialization transients)."""
  skip_final_steps: int = 0
  """Number of final steps to skip (to remove end transients)."""
  data_source: Literal["joint_pos", "actions_obs", "policy_output"] = "joint_pos"
  """Data source: 'joint_pos' (actual positions), 'actions_obs' (previous actions from obs), 'policy_output' (fresh policy commands - smoothest)."""
  compare_csv: str | None = None
  """Path to imitation CSV file to compare with policy outputs (optional)."""


def _apply_play_env_overrides(cfg: ManagerBasedRlEnvCfg, forward_speed: float = 0.8) -> None:
  """Apply PLAY mode overrides - copied from play.py."""
  cfg.episode_length_s = int(1e9)

  assert "policy" in cfg.observations
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  cfg.events.pop("push_robot", None)

  assert cfg.scene.terrain is not None
  terrain_gen = cfg.scene.terrain.terrain_generator
  if terrain_gen is not None:
    terrain_gen.curriculum = False
    terrain_gen.num_cols = 5
    terrain_gen.num_rows = 5
    terrain_gen.border_width = 10.0

  # Disable terminations (except timeout) to prevent episode resets
  if cfg.terminations is not None:
    # Keep only time_out termination
    terminations_to_remove = [key for key in cfg.terminations.keys() if key != "time_out"]
    for key in terminations_to_remove:
      cfg.terminations.pop(key, None)

  # Set fixed velocity command for straight walking.
  if cfg.commands is not None and "twist" in cfg.commands:
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Set fixed straight forward walking at specified speed, no lateral, no turning
    twist_cmd.ranges.lin_vel_x = (forward_speed, forward_speed)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    # Disable command resampling by setting very long resampling time
    twist_cmd.resampling_time_range = (1e9, 1e9)
    print(f"[INFO]: Set forward walking speed to {forward_speed} m/s (no resampling)")


def run_analysis(task: str, cfg: AnalyzeConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configs - same as play.py
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
  _apply_play_env_overrides(env_cfg, forward_speed=cfg.forward_speed)

  agent_cfg = load_cfg_from_registry(task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  # Check if tracking task
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  # Load checkpoint
  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  log_dir = resume_path.parent

  # Set num_envs
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  # Create environment - no render_mode (headless)
  env = gym.make(task, cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Load policy - same as play.py
  if is_tracking_task:
    runner = MotionTrackingOnPolicyRunner(
      env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )
  else:
    runner = OnPolicyRunner(
      env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO]: Loaded checkpoint: {resume_path.name}")
  print(f"[INFO]: Running for {cfg.num_steps} steps with {cfg.num_envs} environment(s)")

  # Get robot entity
  robot = env.unwrapped.scene["robot"]
  joint_names = robot.joint_names
  num_joints = len(joint_names)

  print(f"[INFO]: Tracking {num_joints} joints: {joint_names}")

  # Filter out toe joints (passive/unactuated)
  actuated_joint_indices = [i for i, name in enumerate(joint_names) if "toe" not in name]
  actuated_joint_names = [joint_names[i] for i in actuated_joint_indices]
  num_actuated_joints = len(actuated_joint_names)

  print(f"[INFO]: Filtering to {num_actuated_joints} actuated joints (excluding toe joints)")
  print(f"[INFO]: Actuated joints: {actuated_joint_names}")

  # Get control frequency from environment
  env_unwrapped = env.unwrapped
  if hasattr(env_unwrapped, 'step_dt'):
    control_dt = env_unwrapped.step_dt
  elif hasattr(env_unwrapped, 'dt'):
    control_dt = env_unwrapped.dt
  else:
    # Default assumption for MuJoCo RL environments
    control_dt = 0.02  # 50Hz default

  control_freq = 1.0 / control_dt
  print(f"[INFO]: Environment control frequency: {control_freq:.1f} Hz (dt={control_dt:.4f}s)")
  print(f"[INFO]: Recording at EVERY timestep (no decimation) - same as Ariel's logging approach")

  # If viewer is requested, run viewer and skip analysis
  if cfg.viewer != "none":
    print(f"[INFO]: Running viewer mode (--viewer {cfg.viewer})")
    print("[INFO]: No data will be collected. To collect data, run without --viewer flag.")

    # Handle "auto" viewer selection
    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    print(f"[INFO]: Launching {resolved_viewer} viewer...")
    if resolved_viewer == "native":
      NativeMujocoViewer(cast(EnvProtocol, env), policy).run()
    elif resolved_viewer == "viser":
      ViserViewer(cast(EnvProtocol, env), policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()
    return

  # Storage for joint positions
  joint_positions_history = []

  # Reset environment
  obs, _ = env.reset()

  # Get observation structure dynamically from the observation manager (like Ariel's code)
  obs_manager = env.unwrapped.observation_manager
  obs_group = "policy"  # The observation group we're using

  # Extract term names and dimensions dynamically
  obs_term_names = obs_manager._group_obs_term_names[obs_group]
  obs_term_dims = obs_manager._group_obs_term_dim[obs_group]

  # Create a mapping of term names to their index ranges
  obs_term_indices = {}
  current_idx = 0
  for term_name, term_dim in zip(obs_term_names, obs_term_dims):
    term_size = int(torch.tensor(term_dim).prod())  # Handle multi-dimensional terms
    obs_term_indices[term_name] = (current_idx, current_idx + term_size)
    current_idx += term_size

  print(f"[INFO]: Observation structure for group '{obs_group}':")
  for term_name, (start_idx, end_idx) in obs_term_indices.items():
    print(f"  {term_name}: indices [{start_idx}:{end_idx}]")

  # Determine which observation term to extract based on data_source
  if cfg.data_source == "policy_output":
    data_indices = None  # Will extract from policy output directly
    print("[INFO]: Extracting POLICY OUTPUT (fresh policy commands)")
  elif cfg.data_source == "actions_obs":
    if "actions" in obs_term_indices:
      data_indices = obs_term_indices["actions"]
      print(f"[INFO]: Extracting ACTIONS from observation indices {data_indices[0]}:{data_indices[1]} (previous actions)")
    else:
      raise ValueError("'actions' term not found in observations")
  else:  # joint_pos
    if "joint_pos" in obs_term_indices:
      data_indices = obs_term_indices["joint_pos"]
      print(f"[INFO]: Extracting JOINT_POS from observation indices {data_indices[0]}:{data_indices[1]} (actual positions)")
    else:
      raise ValueError("'joint_pos' term not found in observations")

  print(f"[INFO]: Data source: {cfg.data_source}")

  # Collect data
  print("[INFO]: Collecting data (headless mode)...")
  for step in range(cfg.num_steps):
    if step % 100 == 0:
      print(f"  Step {step}/{cfg.num_steps} (recording {len(joint_positions_history)} samples)")

    # Get action from policy
    with torch.no_grad():
      action = policy(obs)

    # Record at EVERY timestep (no decimation - same as Ariel's approach)
    if cfg.data_source == "policy_output":
      # Extract fresh policy output (smoothest - what the policy just commanded)
      data_to_log = action.clone().cpu().numpy()
      joint_positions_history.append(data_to_log)
    else:
      # Extract data from observation (either joint_pos or previous actions)
      # Handle TensorDict, dict, or direct tensor
      if hasattr(obs, 'get'):
        policy_obs = obs.get("policy", obs)
      else:
        policy_obs = obs

      # Extract data from observation
      if isinstance(policy_obs, torch.Tensor):
        # Extract data based on configured source
        data_from_obs = policy_obs[:, data_indices[0]:data_indices[1]].clone().cpu().numpy()
        joint_positions_history.append(data_from_obs)
      else:
        raise RuntimeError(
          f"Unexpected policy observation type: {type(policy_obs)}. "
          f"Expected torch.Tensor but got {type(policy_obs)}. "
          f"Full observation type: {type(obs)}"
        )

    # Step environment
    obs, _, _, _ = env.step(action)

  env.close()

  print(f"[INFO]: Data collection complete. Processing {cfg.data_source} absolute positions...")

  # Convert to numpy array: (num_recorded_steps, num_envs, num_actuated_joints)
  # Note: We already filtered to actuated joints during data collection
  joint_positions_history = np.array(joint_positions_history)

  print(f"[INFO]: Collected {joint_positions_history.shape[0]} samples at {control_freq:.1f} Hz (every timestep)")
  print(f"[INFO]: Data source: {cfg.data_source}")

  # Flatten absolute positions
  joint_positions_flat = joint_positions_history.reshape(-1, num_actuated_joints)

  print(f"[INFO]: Processed {joint_positions_flat.shape[0]} joint position samples")

  # Create output directory
  output_path = Path(cfg.output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # Save raw data
  np.savez(
    output_path / "joint_positions.npz",
    joint_positions=joint_positions_flat,
    joint_names=actuated_joint_names,
  )
  print(f"[INFO]: Saved raw data to {output_path / 'joint_positions.npz'}")

  # Compute and save statistics (using all data including transients for reference)
  print("[INFO]: Computing statistics...")
  stats = []
  for i, joint_name in enumerate(actuated_joint_names):
    positions = joint_positions_flat[:, i]
    stats.append({
      "joint": joint_name,
      "mean": float(np.mean(positions)),
      "std": float(np.std(positions)),
      "min": float(np.min(positions)),
      "max": float(np.max(positions)),
      "median": float(np.median(positions)),
    })

  print(f"[INFO]: Note - plots will skip first {cfg.skip_initial_steps} and last {cfg.skip_final_steps} timesteps to show steady-state walking")

  # Print statistics table
  print("\n" + "=" * 80)
  print("JOINT POSITION STATISTICS (ABSOLUTE)")
  print("=" * 80)
  print(f"{'Joint':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
  print("-" * 80)
  for stat in stats:
    print(
      f"{stat['joint']:<30} "
      f"{stat['mean']:>10.6f} "
      f"{stat['std']:>10.6f} "
      f"{stat['min']:>10.6f} "
      f"{stat['max']:>10.6f} "
      f"{stat['median']:>10.6f}"
    )
  print("=" * 80)

  # Save statistics to CSV
  import csv

  with open(output_path / "joint_position_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["joint", "mean", "std", "min", "max", "median"])
    writer.writeheader()
    writer.writerows(stats)

  print(f"[INFO]: Saved statistics to {output_path / 'joint_position_stats.csv'}")

  # Generate visualizations (optional - requires matplotlib)
  try:
    import matplotlib.pyplot as plt

    # 1. Time series plot of joint positions
    print("[INFO]: Generating joint position time series...")

    # Determine slice range (skip transients)
    start_idx = cfg.skip_initial_steps
    end_idx = len(joint_positions_history) - cfg.skip_final_steps
    if end_idx <= start_idx:
      print(f"[WARN]: Not enough data after skipping transients. Using all data.")
      start_idx = 0
      end_idx = len(joint_positions_history)
    else:
      print(f"[INFO]: Plotting steady-state data from timestep {start_idx} to {end_idx}")

    num_cols = 1
    num_rows = num_actuated_joints
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 2.5 * num_rows))
    axes = axes.flatten() if num_actuated_joints > 1 else [axes]

    for i, joint_name in enumerate(actuated_joint_names):
      # Get positions for this joint, skipping transients (first env only)
      positions = joint_positions_history[start_idx:end_idx, 0, i]

      ax = axes[i]
      ax.plot(positions, linewidth=0.8)
      ax.set_ylabel(f"{joint_name.replace('_joint', '')}", fontsize=9)
      ax.grid(True, alpha=0.3)

      # Only show x-label on bottom plot
      if i == num_actuated_joints - 1:
        ax.set_xlabel("timestep")
      else:
        ax.set_xticklabels([])

    # Add title to top subplot
    axes[0].set_title("joint_pos", fontsize=10, loc='left')

    plt.tight_layout()
    plt.savefig(output_path / "joint_positions_timeseries.png", dpi=150)
    print(f"[INFO]: Saved time series to {output_path / 'joint_positions_timeseries.png'}")
    plt.close()

    # 2. Histograms of joint positions (using steady-state data only)
    print("[INFO]: Generating joint position histograms...")

    # Get absolute positions from steady-state portion only
    steady_state_positions = joint_positions_history[start_idx:end_idx, :, :]
    steady_state_positions_flat = steady_state_positions.reshape(-1, num_actuated_joints)

    num_cols = 3
    num_rows = (num_actuated_joints + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten() if num_actuated_joints > 1 else [axes]

    for i, joint_name in enumerate(actuated_joint_names):
      positions = steady_state_positions_flat[:, i]
      mean_pos = np.mean(positions)
      std_pos = np.std(positions)

      ax = axes[i]
      ax.hist(positions, bins=cfg.bins, alpha=0.7, edgecolor="black")
      ax.set_title(f"{joint_name}\nμ={mean_pos:.4f}, σ={std_pos:.4f}")
      ax.set_xlabel("Joint Position (rad)")
      ax.set_ylabel("Frequency")
      ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_actuated_joints, len(axes)):
      axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "joint_positions_histogram.png", dpi=150)
    print(f"[INFO]: Saved histogram to {output_path / 'joint_positions_histogram.png'}")
    plt.close()

    # 3. Comparison with imitation CSV (if provided)
    if cfg.compare_csv is not None:
      print(f"[INFO]: Generating comparison with imitation data from {cfg.compare_csv}...")

      # Load CSV
      import pandas as pd
      imitation_df = pd.read_csv(cfg.compare_csv)

      # CSV column order: left_hip_pitch, right_hip_pitch, left_hip_roll, right_hip_roll,
      #                   left_hip_yaw, right_hip_yaw, left_knee, right_knee,
      #                   left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll
      # Our order: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
      #            right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll

      # Reorder columns to match our joint order
      csv_to_our_order = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
      imitation_data = imitation_df.iloc[:, csv_to_our_order].values  # (num_samples, 12)

      # Truncate to same length for comparison
      min_len = min(len(imitation_data), len(joint_positions_history))
      imitation_data = imitation_data[:min_len]
      policy_data = joint_positions_history[:min_len, 0, :]  # (min_len, 12)

      print(f"[INFO]: Comparing {min_len} timesteps")
      print(f"[INFO]: Imitation data shape: {imitation_data.shape}")
      print(f"[INFO]: Policy data shape: {policy_data.shape}")

      # Plot comparison
      num_cols = 1
      num_rows = num_actuated_joints
      fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2.5 * num_rows))
      axes = axes.flatten() if num_actuated_joints > 1 else [axes]

      for i, joint_name in enumerate(actuated_joint_names):
        ax = axes[i]

        # Plot both datasets
        ax.plot(imitation_data[:, i], linewidth=0.8, alpha=0.7, label='Imitation', color='blue')
        ax.plot(policy_data[:, i], linewidth=0.8, alpha=0.7, label='Policy', color='orange')

        # Show ranges
        imitation_range = (np.min(imitation_data[:, i]), np.max(imitation_data[:, i]))
        policy_range = (np.min(policy_data[:, i]), np.max(policy_data[:, i]))

        ax.set_ylabel(f"{joint_name.replace('_joint', '')}\n"
                      f"Imit: [{imitation_range[0]:.2f}, {imitation_range[1]:.2f}]\n"
                      f"Pol: [{policy_range[0]:.2f}, {policy_range[1]:.2f}]",
                      fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=6)

        # Only show x-label on bottom plot
        if i == num_actuated_joints - 1:
          ax.set_xlabel("timestep")
        else:
          ax.set_xticklabels([])

      axes[0].set_title("Imitation vs Policy Comparison", fontsize=10, loc='left')

      plt.tight_layout()
      plt.savefig(output_path / "imitation_vs_policy_comparison.png", dpi=150)
      print(f"[INFO]: Saved comparison to {output_path / 'imitation_vs_policy_comparison.png'}")
      plt.close()

      # Print range comparison table
      print("\n" + "=" * 100)
      print("RANGE COMPARISON: Imitation vs Policy")
      print("=" * 100)
      print(f"{'Joint':<30} {'Imit Min':>12} {'Imit Max':>12} {'Pol Min':>12} {'Pol Max':>12} {'Match?':>8}")
      print("-" * 100)
      for i, joint_name in enumerate(actuated_joint_names):
        imitation_min = np.min(imitation_data[:, i])
        imitation_max = np.max(imitation_data[:, i])
        policy_min = np.min(policy_data[:, i])
        policy_max = np.max(policy_data[:, i])

        # Check if ranges are similar (within 50% tolerance)
        range_match = abs(imitation_max - imitation_min - (policy_max - policy_min)) / (imitation_max - imitation_min + 1e-6) < 0.5
        match_str = "✓" if range_match else "✗"

        print(
          f"{joint_name:<30} "
          f"{imitation_min:>12.4f} "
          f"{imitation_max:>12.4f} "
          f"{policy_min:>12.4f} "
          f"{policy_max:>12.4f} "
          f"{match_str:>8}"
        )
      print("=" * 100)

  except ImportError:
    print("[WARN]: matplotlib not available, skipping visualization generation")
    print("[WARN]: Install with: uv pip install matplotlib")

  print(f"\n[SUCCESS]: Analysis complete! Results saved to {output_path}")


def main():
  # Parse task name - same as play.py
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse remaining arguments
  args = tyro.cli(
    AnalyzeConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
  )
  del remaining_args

  run_analysis(chosen_task, args)


if __name__ == "__main__":
  main()
