import torch


from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, AssetBaseCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.utils.timer import Timer, TIMER_CUMULATIVE
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.commands import GoalCommand

from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.utils import (
    get_robot_pos,
    get_robot_quat,
    get_robot_lin_vel_w,
    get_robot_rot_vel_w,
)


##
# - lidar
##
def lidar_obs_dist_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(
        sensor.data.ray_hits_w[..., :2] - sensor.data.pos_w[..., :2].unsqueeze(1), dim=2
    )

    return distances


def lidar_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]

    height_diffs = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[..., 2].unsqueeze(1)

    return height_diffs


# def lidar_obs_dist_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     sensor: SensorBase = env.scene.sensors[sensor_cfg.name]

#     # Extract and ensure tensors are contiguous
#     ray_hits_w = sensor.data.ray_hits_w  # [..., :2]  # Shape: (N, 36, 2)
#     pos_w = sensor.data.pos_w  # [..., :2]  # Shape: (N, 2)

#     # Use broadcasting without unsqueeze
#     diff = ray_hits_w - pos_w[:, None, :]  # Shape: (N, 36, 2)

#     # Compute distances
#     distances = torch.norm(diff, dim=2)  # Shape: (N, 36)

#     return distances


def lidar_obs_dist_2d_log(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=2)

    return torch.log(distances + 1e-6)


##
# - positions
##
def pose_2d_env(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the pose of the entity relative to the terrain origin.
    x,y position and heading in the form of cos(theta), sin(theta)."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = get_robot_pos(entity)
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - heading
    quat = get_robot_quat(entity).squeeze(1)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    cos_yaw, sin_yaw = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)

    pose_2d = torch.cat([rel_pos[:, :2], cos_yaw, sin_yaw], dim=-1)
    return pose_2d


def pose_3d_env(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position and Quaternion in environment frame"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = get_robot_pos(entity)
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - quaternion
    quat = get_robot_quat(entity).squeeze(1)

    return torch.cat([rel_pos, quat], dim=-1)


def box_pose(env: ManagerBasedEnv, entity_str: str, pov_entity: SceneEntityCfg) -> torch.Tensor:
    """Returns the pose of all entities relative to the terrain origin.
    x,y position and heading in the form of cos(theta), sin(theta)."""

    # - box poses
    box_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if entity_str in asset]
    box_poses = []
    box_quats = []

    for box_id in box_ids:
        box_poses.append(env.scene.rigid_objects[box_id].data.root_pos_w)
        box_quats.append(env.scene.rigid_objects[box_id].data.root_quat_w)

    boxes_positions_w = torch.stack(box_poses, dim=1)
    boxes_quats_w = torch.stack(box_quats, dim=1)

    # - robot pose
    robot = env.scene[pov_entity.name]
    robot_pos_w = get_robot_pos(robot)
    robot_quat_w = get_robot_quat(robot)

    # Expand robot pose to match the number of boxes
    robot_pos_w_expanded = robot_pos_w.unsqueeze(1).expand_as(boxes_positions_w)
    robot_quat_w_expanded = robot_quat_w.unsqueeze(1).expand_as(boxes_quats_w)

    # - calculate pose of boxes in robot frame
    t_box_robot, q_box_robot = math_utils.subtract_frame_transforms(
        robot_pos_w_expanded, robot_quat_w_expanded, boxes_positions_w, boxes_quats_w
    )

    # Extract x, y positions
    x = t_box_robot[..., 0]
    y = t_box_robot[..., 1]

    # Compute yaw angle from the quaternion
    w = q_box_robot[..., 0]
    x_q = q_box_robot[..., 1]
    y_q = q_box_robot[..., 2]
    z_q = q_box_robot[..., 3]

    # Yaw angle computation
    sin_yaw = 2 * (w * z_q + x_q * y_q)
    cos_yaw = 1 - 2 * (y_q * y_q + z_q * z_q)

    # yaw = torch.atan2(sin_yaw, cos_yaw)
    # Stack the results into a single tensor
    pose = torch.stack([x, y, cos_yaw, sin_yaw], dim=-1)

    return pose


##
# - velocities
##
def velocity_2d_b(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, pov_entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity vector of the entity rotated to the robot's frame (only yaw considered).
    The robots velocity is neglected."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    robot: RigidObject | Articulation = env.scene[pov_entity_cfg.name]

    robot_quat_w = math_utils.yaw_quat(get_robot_quat(robot))
    if entity == robot:
        lin_vel_w = get_robot_lin_vel_w(robot)
        lin_vel_b = math_utils.quat_rotate_inverse(robot_quat_w, lin_vel_w)
        ang_vel_z_w = get_robot_rot_vel_w(robot)[..., 2]
        return torch.cat([lin_vel_b, ang_vel_z_w.unsqueeze(1)], dim=-1)

    entity_vel_w = entity.data.body_lin_vel_w.squeeze(1)
    entity_ang_vel_z = entity.data.body_ang_vel_w.squeeze(1)[..., 2]
    entity_vel_b = math_utils.quat_rotate_inverse(robot_quat_w, entity_vel_w)
    return torch.cat([entity_vel_b[..., :2], entity_ang_vel_z.unsqueeze(1)], dim=-1)


def rotation_velocity_2d_b(
    env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, pov_entity_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Returns the angular velocity in z direction (yaw rotation)"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    robot: RigidObject | Articulation = env.scene[pov_entity_cfg.name]

    if entity == robot:
        ang_vel_z_w = get_robot_rot_vel_w(robot)[..., 2]
        return ang_vel_z_w.unsqueeze(1)

    entity_ang_vel_z = entity.data.body_ang_vel_w.squeeze(1)[..., 2]
    return entity_ang_vel_z.unsqueeze(1)


def velocity_2d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity vector of the entity in the terrain frame."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    entity_vel_w = get_robot_lin_vel_w(entity)
    entity_ang_vel_z = get_robot_rot_vel_w(entity)[..., 2]
    return torch.cat([entity_vel_w[..., :2], entity_ang_vel_z.unsqueeze(1)], dim=-1)


def velocity_3d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the linear and angular velocity vector in the world frame"""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    entity_vel_w = get_robot_lin_vel_w(entity)
    entity_angvel_w = get_robot_rot_vel_w(entity)
    return torch.cat([entity_vel_w, entity_angvel_w], dim=-1)


##
# - goal
##


def dist_to_goal(env: ManagerBasedRLEnv, entity_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Returns the distance to the goal for the given entity."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]

    entity_pos = get_robot_pos(entity)
    goal_pos = goal_cmd_geneator.goal_pos_w

    diff = torch.linalg.norm(entity_pos - goal_pos, dim=-1).unsqueeze(1)
    return diff
