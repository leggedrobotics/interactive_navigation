import gymnasium as gym

from . import agents, rough_env_cfg
from interactive_navigation.tasks.autocurricula_games.jump_boxes.articulation_env_cfg import BoxStairEnvCfg
from interactive_navigation.tasks.autocurricula_games.jump_boxes.rigid_robot_env_cfg import RigidRobotEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-BoxStair-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoxStairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoxStairPPORunnerCfg",
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
