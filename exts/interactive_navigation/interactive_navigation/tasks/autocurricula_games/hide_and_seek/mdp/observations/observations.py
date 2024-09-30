import torch


from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, AssetBaseCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase
from omni.isaac.lab.utils import math as math_utils


def lidar_obs_dist(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=2)

    return distances


def pose_2d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
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


def velocity_2d_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the velocity of the entity relative to the terrain origin.
    x,y velocity and heading in the form of cos(theta), sin(theta)."""
    entity: RigidObject | Articulation = env.scene[entity_cfg.name]

    # - velocity
    vel = entity.data.body_lin_vel_w.squeeze(1)

    # - angular velocity
    ang_vel = entity.data.body_ang_vel_w.squeeze(1)

    velocity_2d = torch.cat([vel[:, :2], ang_vel[:, 2].unsqueeze(1)], dim=-1)

    return velocity_2d
