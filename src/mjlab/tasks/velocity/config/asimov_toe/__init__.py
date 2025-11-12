import gymnasium as gym

from .env_cfgs import ASIMOV_FLAT_ENV_CFG, ASIMOV_ROUGH_ENV_CFG

gym.register(
  id="Mjlab-Velocity-Rough-Asimov-Toe",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_ROUGH_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Asimov-Toe",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)
