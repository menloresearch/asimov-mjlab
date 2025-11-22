import gymnasium as gym

from .env_cfgs import ASIMOV_FLAT_ENV_CFG, ASIMOV_ROUGH_ENV_CFG
from .env_cfgs_learned import ASIMOV_FLAT_ENV_CFG_LEARNED, ASIMOV_ROUGH_ENV_CFG_LEARNED

# Original configurations with ideal actuators
gym.register(
  id="Mjlab-Velocity-Rough-Asimov",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_ROUGH_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Asimov",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)

# New configurations with learned actuator dynamics
gym.register(
  id="Mjlab-Velocity-Rough-Asimov-Learned",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_ROUGH_ENV_CFG_LEARNED,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Asimov-Learned",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": ASIMOV_FLAT_ENV_CFG_LEARNED,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AsimovPPORunnerCfg",
  },
)
