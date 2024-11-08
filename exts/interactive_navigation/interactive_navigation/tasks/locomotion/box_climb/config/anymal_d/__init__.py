import gymnasium as gym

from . import agents, step_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-BoxStep-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": step_env_cfg.AnymalDBoxClimbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDBoxClimbPPORunnerCfg",
    },
)
