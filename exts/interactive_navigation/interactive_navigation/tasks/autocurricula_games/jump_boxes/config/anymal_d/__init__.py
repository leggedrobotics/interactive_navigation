"""Not supported yet."""

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg


##
# Register Gym environments.
##


import gymnasium as gym

from . import agents, rough_env_cfg
from interactive_navigation.tasks.autocurricula_games.jump_boxes.anymal_d_box_stair_env import AnymalBoxeStairEnvCfg
from interactive_navigation.tasks.autocurricula_games.jump_boxes.rigid_robot_env_cfg import RigidRobotEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-BoxStairs-Anymal-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalBoxeStairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalBoxeStairPPORunnerCfg",
    },
)

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
