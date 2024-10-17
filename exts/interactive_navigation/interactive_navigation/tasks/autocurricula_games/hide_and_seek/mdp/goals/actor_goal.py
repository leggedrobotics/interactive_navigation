import torch


from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, AssetBaseCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.utils.timer import Timer, TIMER_CUMULATIVE


def generated_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the generated goal for the given command.
    Note, the command needs to implement the goal property."""
    return env.command_manager._terms[command_name].goal  # type: ignore


# def robot_heigh_goal(env: ManagerBasedEnv, goal_height: int) -> torch.Tensor:
#     """Returns the current goal height for the robot. Note that here we do not need access to the current state,
#     since this is the goal for conditioning."""
#     # TODO: goal dependent on env/terrain. Is it then still single goal exploration???

#     goal_height_tensor = torch.zeros(env.num_envs, 1, device=env.device) + goal_height

#     return goal_height_tensor
