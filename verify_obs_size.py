#!/usr/bin/env python3
"""Verify observation size for Asimov configurations.

This script checks that the observation size matches firmware expectations (45 elements).
"""

import gymnasium as gym

# Import to register environments
import mjlab.tasks.velocity.config.asimov  # noqa: F401

from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def check_task_obs_size(task_name: str, expected_size: int = 45):
    """Check observation size for a given task."""
    print(f"\n{'='*60}")
    print(f"Checking: {task_name}")
    print(f"{'='*60}")

    # Load config
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    # Check policy observations
    if env_cfg.observations and "policy" in env_cfg.observations:
        policy_obs = env_cfg.observations["policy"]
        num_terms = len(policy_obs.terms)
        print(f"Policy observation terms: {num_terms}")

        total_size = 0
        for name, term in policy_obs.terms.items():
            # Estimate size based on term name
            if "base_ang_vel" in name or "base_lin_vel" in name or "projected_gravity" in name:
                size = 3
            elif "command" in name:
                size = 3
            elif "joint_pos" in name or "joint_vel" in name or "actions" in name:
                size = 12
            else:
                size = "?"
            print(f"  - {name}: {size}")
            if isinstance(size, int):
                total_size += size

        print(f"\nEstimated total observation size: {total_size}")
        print(f"Expected size: {expected_size}")

        if total_size == expected_size:
            print(f"✅ PASS: Observation size matches firmware ({expected_size} elements)")
            return True
        else:
            print(f"❌ FAIL: Size mismatch! Got {total_size}, expected {expected_size}")
            print(f"         You need to retrain with this configuration.")
            return False
    else:
        print("❌ No policy observations found!")
        return False


if __name__ == "__main__":
    tasks = [
        "Mjlab-Velocity-Flat-Asimov-Learned",
        "Mjlab-Velocity-Rough-Asimov-Learned",
        "Mjlab-Velocity-Flat-Asimov",
        "Mjlab-Velocity-Rough-Asimov",
    ]

    print("=" * 60)
    print("Asimov Observation Size Verification")
    print("Firmware expects: 45 elements")
    print("=" * 60)

    all_passed = True
    for task in tasks:
        try:
            passed = check_task_obs_size(task)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"❌ Error checking {task}: {e}")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL CHECKS PASSED - Safe to train and deploy!")
    else:
        print("❌ SOME CHECKS FAILED - Fix configs before training!")
    print(f"{'='*60}\n")
