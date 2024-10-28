import torch
from omni.isaac.lab.assets import Articulation, RigidObject


def get_robot_pos(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the position of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_pos_w[:, -1, :]
    else:
        return robot.data.root_pos_w


def get_robot_quat(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the quaternion of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_quat_w[:, -1, :]
    else:
        return robot.data.root_quat_w


def get_robot_lin_vel_w(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the linear velocity of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_lin_vel_w[:, -1, :]
    else:
        return robot.data.root_lin_vel_w


def get_robot_rot_vel_w(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the rotational velocity of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_ang_vel_w[:, -1, :]
    else:
        return robot.data.root_ang_vel_w
