"""Not supported yet."""

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.anymal_env_cfg import CrlAnymalEnvCfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.anymal_metra_env_cfg import MetraAnymalEnvCfg


##
# CRL
##

gym.register(
    id="Isaac-CRL-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrlAnymalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_crl_cfg:AnymalTestCrlRunnerCfg",
    },
)


gym.register(
    id="Isaac-METRA-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MetraAnymalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_metra_cfg:AnymalMetraPPORunnerCfg",
    },
)


##
# Register Gym environments.
##

# gym.register(
#     id="Template-Isaac-Velocity-Flat-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.AnymalDFlatEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Template-Isaac-Velocity-Flat-Anymal-D-Play-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.AnymalDFlatEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Template-Isaac-Velocity-Rough-Anymal-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": rough_env_cfg.AnymalDRoughEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
#     },
# )

gym.register(
    id="Isaac-Games-HideAndSeek-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AnymalDRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
    },
)
