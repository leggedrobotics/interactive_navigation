import torch


from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, AssetBaseCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.utils.timer import Timer, TIMER_CUMULATIVE


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


def pose_2d_to(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the pose of the entity relative to the terrain origin.
    x,y position and heading in the form of cos(theta), sin(theta)."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - position
    pos = entity.data.body_pos_w
    terrain = env.scene.terrain
    terrain_origins = terrain.env_origins
    rel_pos = pos.squeeze(1) - terrain_origins

    # - heading
    quat = entity.data.body_quat_w.squeeze(1)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    cos_yaw, sin_yaw = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)

    pose_2d = torch.cat([rel_pos[:, :2], cos_yaw, sin_yaw], dim=-1)
    return pose_2d


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
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

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


def velocity_2d_b(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg, pov_entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity vector of the entity rotated to the robot's frame (only yaw considered).
    The robots velocity is neglected."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]
    robot: RigidObject | Articulation = env.scene[pov_entity_cfg.name]

    if entity == robot:
        lin_vel = entity.data.root_lin_vel_b[..., :2]
        ang_vel_z = entity.data.root_ang_vel_b[..., 2]
        return torch.cat([lin_vel, ang_vel_z.unsqueeze(1)], dim=-1)

    entity_vel_w = entity.data.body_lin_vel_w.squeeze(1)
    entity_ang_vel_z = entity.data.body_ang_vel_w.squeeze(1)[..., 2]
    robot_quat_w = math_utils.yaw_quat(robot.data.root_quat_w)
    entity_vel_b = math_utils.quat_rotate_inverse(robot_quat_w, entity_vel_w)
    return torch.cat([entity_vel_b[..., :2], entity_ang_vel_z.unsqueeze(1)], dim=-1)
