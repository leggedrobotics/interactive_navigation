import torch


from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, RayCaster, SensorBase


def lidar_obs_dist(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """lidar scan from the given sensor w.r.t. the sensor's frame."""
    # TODO calculate the proper distances
    sensor: SensorBase = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.vector_norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=2)

    return distances
