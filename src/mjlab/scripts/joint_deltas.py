"""Script to analyze joint deltas from a trained policy - simplified version based on play.py."""

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
  output_dir: str = "./joint_delta_analysis"
  """Directory to save histograms and data."""
  bins: int = 50
  """Number of bins for histograms."""
  viewer: Literal["none", "auto", "native", "viser"] = "none"
  """Viewer backend: none (headless analysis), auto (detect display), native (mujoco), viser (web-based)."""
  forward_speed: float = 0.8
  """Forward walking speed in m/s (default: 0.8)."""
  skip_initial_steps: int = 200
  """Number of initial steps to skip (to remove initialization transients)."""
  skip_final_steps: int = 200
  """Number of final steps to skip (to remove end transients)."""


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

  # Collect data
  print("[INFO]: Collecting data (headless mode)...")
  for step in range(cfg.num_steps):
    if step % 100 == 0:
      print(f"  Step {step}/{cfg.num_steps}")

    # Get current joint positions
    joint_pos = robot.data.joint_pos.clone().cpu().numpy()
    joint_positions_history.append(joint_pos)

    # Get action from policy
    with torch.no_grad():
      action = policy(obs)

    # Step environment
    obs, _, _, _ = env.step(action)

  env.close()

  print("[INFO]: Data collection complete. Computing joint deltas...")

  # Convert to numpy array: (num_steps, num_envs, num_joints)
  joint_positions_history = np.array(joint_positions_history)

  # Filter to only actuated joints
  joint_positions_history = joint_positions_history[:, :, actuated_joint_indices]

  # Compute joint deltas
  joint_deltas = np.diff(joint_positions_history, axis=0)
  joint_deltas_flat = joint_deltas.reshape(-1, num_actuated_joints)

  print(f"[INFO]: Computed {joint_deltas_flat.shape[0]} joint delta samples")

  # Create output directory
  output_path = Path(cfg.output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # Save raw data
  np.savez(
    output_path / "joint_deltas.npz",
    joint_deltas=joint_deltas_flat,
    joint_names=actuated_joint_names,
  )
  print(f"[INFO]: Saved raw data to {output_path / 'joint_deltas.npz'}")

  # Compute and save statistics (using all data including transients for reference)
  print("[INFO]: Computing statistics...")
  stats = []
  for i, joint_name in enumerate(actuated_joint_names):
    deltas = joint_deltas_flat[:, i]
    stats.append({
      "joint": joint_name,
      "mean": float(np.mean(deltas)),
      "std": float(np.std(deltas)),
      "min": float(np.min(deltas)),
      "max": float(np.max(deltas)),
      "median": float(np.median(deltas)),
    })

  print(f"[INFO]: Note - plots will skip first {cfg.skip_initial_steps} and last {cfg.skip_final_steps} timesteps to show steady-state walking")

  # Print statistics table
  print("\n" + "=" * 80)
  print("JOINT DELTA STATISTICS")
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

  with open(output_path / "joint_delta_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["joint", "mean", "std", "min", "max", "median"])
    writer.writeheader()
    writer.writerows(stats)

  print(f"[INFO]: Saved statistics to {output_path / 'joint_delta_stats.csv'}")

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

    # 2. Histograms of joint deltas (using steady-state data only)
    print("[INFO]: Generating joint delta histograms...")

    # Compute deltas from steady-state portion only
    steady_state_positions = joint_positions_history[start_idx:end_idx, :, :]
    steady_state_deltas = np.diff(steady_state_positions, axis=0)
    steady_state_deltas_flat = steady_state_deltas.reshape(-1, num_actuated_joints)

    num_cols = 3
    num_rows = (num_actuated_joints + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten() if num_actuated_joints > 1 else [axes]

    for i, joint_name in enumerate(actuated_joint_names):
      deltas = steady_state_deltas_flat[:, i]
      mean_delta = np.mean(deltas)
      std_delta = np.std(deltas)

      ax = axes[i]
      ax.hist(deltas, bins=cfg.bins, alpha=0.7, edgecolor="black")
      ax.set_title(f"{joint_name}\nμ={mean_delta:.4f}, σ={std_delta:.4f}")
      ax.set_xlabel("Joint Delta (rad)")
      ax.set_ylabel("Frequency")
      ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_actuated_joints, len(axes)):
      axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "joint_deltas_histogram.png", dpi=150)
    print(f"[INFO]: Saved histogram to {output_path / 'joint_deltas_histogram.png'}")
    plt.close()

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
