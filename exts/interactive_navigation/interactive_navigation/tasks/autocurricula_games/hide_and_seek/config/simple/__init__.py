import gymnasium as gym

from . import agents, rough_env_cfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.hide_and_seek_env_cfg import HideSeekEnvCfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.rigid_robot_env_cfg import RigidRobotEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Games-HideAndSeek-Simple-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HideSeekEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HideSeekRelationlPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Games-RigidRobot-Simple-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RigidRobotEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HideSeekRelationlPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Games-RigidRobot-Recurrent-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RigidRobotEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RecurrentPPORunnerCfg",
    },
)


# gym.register(
#     id="Isaac-Games-HideAndSeek-Simple-D-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": HideSeekEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HideSeekPPORunnerCfg",
#     },
# )
