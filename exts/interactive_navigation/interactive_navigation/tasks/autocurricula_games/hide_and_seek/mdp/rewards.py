from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def dummy_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Placeholder reward function that returns a dummy reward."""
    asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "asset" in asset]

    asset_poses = []
    for asset_id in asset_ids:
        asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)

    positions = torch.stack(asset_poses)
    # positions is expected to be of shape (N, 2)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape (N, N, 2)
    dist_squared = (diff**2).sum(-1)  # shape (N, N)

    # Convert squared distances to actual distances
    dist = torch.sqrt(dist_squared)

    # Since dist[i, j] and dist[j, i] are the same, we can sum up one triangle and diagonal, then double it.
    # We avoid doubling the diagonal since it is always zero.
    triu_indices = torch.triu_indices(dist.size(0), dist.size(1), offset=1)
    sum_distances = dist[triu_indices[0], triu_indices[1]].sum(0)

    # TODO calculate a reward from this

    return torch.zeros(env.num_envs).to(env.device)


class BoxMovingReward:

    def __init__(self):
        self.prev_dist = None

    def obstacle_to_middle(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
        """Placeholder reward function that returns a dummy reward."""
        terrain_origins = env.scene.terrain.env_origins

        asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "asset" in asset]
        asset_poses = []
        for asset_id in asset_ids:
            asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)
        positions = torch.stack(asset_poses)

        vec_to_center = terrain_origins.unsqueeze(0) - positions
        distances_to_center = torch.linalg.vector_norm(vec_to_center, dim=-1)

        if self.prev_dist is None:
            self.prev_dist = distances_to_center
            return torch.zeros(env.num_envs).to(env.device)

        at_goal = distances_to_center < threshold
        how_much_closer_than_before = self.prev_dist - distances_to_center
        moved_to_far = how_much_closer_than_before > 0.5

        reward = how_much_closer_than_before
        reward = torch.where(at_goal, torch.ones_like(reward), reward)
        reward = torch.where(moved_to_far, torch.zeros_like(reward), reward)

        # reward = -sum_distances / (len(asset_ids) * max(env.scene.terrain.cfg.terrain_generator.size))

        self.prev_dist = distances_to_center
        return torch.sum(reward, dim=0)
