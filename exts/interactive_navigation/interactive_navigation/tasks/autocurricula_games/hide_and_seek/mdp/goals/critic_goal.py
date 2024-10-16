import torch


from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, AssetBaseCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.utils.timer import Timer, TIMER_CUMULATIVE


def robot_heigh_goal(env: ManagerBasedEnv, scene_entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the current  height for the robot. Note that here we do need access to the current state,
    since this will be used for training"""
    # TODO: goal dependent on terrain???
    robot_pos = get_robot_pos(env.scene[scene_entity_cfg.name])

    goal_height_tensor = robot_pos[:, 2]

    return goal_height_tensor.unsqueeze(1)


##
#  Utility functions
##


def get_robot_pos(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the position of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_pos_w[:, -1, :]
    else:
        return robot.data.root_pos_w
