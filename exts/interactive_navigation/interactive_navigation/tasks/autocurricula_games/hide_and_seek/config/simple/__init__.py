import gymnasium as gym

from . import agents, rough_env_cfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.articulation_env_cfg import MoveUpBoxesEnvCfg
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.rigid_robot_env_cfg import RigidRobotEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Jump-Boxes-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MoveUpBoxesEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JumpeOnBoxesPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Games-RigidRobot-Simple-D-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RigidRobotEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HideSeekRelationalPPORunnerCfg",
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
