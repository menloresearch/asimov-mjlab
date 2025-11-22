from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import ASIMOV_FLAT_ENV_CFG, ASIMOV_ROUGH_ENV_CFG
from .env_cfgs_learned import ASIMOV_FLAT_ENV_CFG_LEARNED, ASIMOV_ROUGH_ENV_CFG_LEARNED
from .rl_cfg import asimov_ppo_runner_cfg

# Original configurations with ideal actuators
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Asimov",
  env_cfg=ASIMOV_ROUGH_ENV_CFG(),
  play_env_cfg=ASIMOV_ROUGH_ENV_CFG(play=True),
  rl_cfg=asimov_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Asimov",
  env_cfg=ASIMOV_FLAT_ENV_CFG(),
  play_env_cfg=ASIMOV_FLAT_ENV_CFG(play=True),
  rl_cfg=asimov_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# New configurations with learned actuator dynamics
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Asimov-Learned",
  env_cfg=ASIMOV_ROUGH_ENV_CFG_LEARNED(),
  play_env_cfg=ASIMOV_ROUGH_ENV_CFG_LEARNED(play=True),
  rl_cfg=asimov_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Asimov-Learned",
  env_cfg=ASIMOV_FLAT_ENV_CFG_LEARNED(),
  play_env_cfg=ASIMOV_FLAT_ENV_CFG_LEARNED(play=True),
  rl_cfg=asimov_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
