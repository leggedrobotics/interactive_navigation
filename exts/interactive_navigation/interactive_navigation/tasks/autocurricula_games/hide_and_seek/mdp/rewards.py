from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def dummy_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Placeholder reward function that returns a dummy reward."""
    asset_ids = ["asset_1", "asset_2", "asset_3"]

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
