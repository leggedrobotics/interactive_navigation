from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal
import functools

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.warp import raycast_mesh
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from omni.isaac.lab.utils.warp import convert_to_warp_mesh
from omni.isaac.lab.terrains.utils import find_flat_patches
from omni.isaac.lab.terrains.terrain_generator_cfg import FlatPatchSamplingCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_root_state_uniform_collision_free(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    max_tries = 100

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    for _ in range(max_tries):

        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        if _is_collision_free(env, asset_cfg, env_ids):
            break

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_multiple_instances_decorator(reset_func: callable) -> callable:
    """Decorator to reset multiple instances of an asset at once."""

    @functools.wraps(reset_func)
    def wrapper(*args, **kwargs):
        asset_configs = kwargs.get("asset_configs", None)
        asset_config = kwargs.get("asset_cfg", None)
        if asset_configs is None and asset_config is None:
            asset_config = SceneEntityCfg("robot")
        if asset_configs is not None and asset_config is not None:
            raise ValueError(
                "The decorator 'reset_multiple_instances_decorator' requires either 'asset_cfg' or 'asset_configs' to be provided, not both."
            )
        if asset_configs is None and asset_config is not None:
            asset_configs = [asset_config]
        for i, asset_cfg in enumerate(asset_configs):
            kwargs["asset_cfg"] = asset_cfg
            kwargs["reset_id"] = i
            reset_func(*args, **kwargs)

    return wrapper


@reset_multiple_instances_decorator
def reset_root_state_uniform_on_terrain_aware(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    offset: list = [0.0, 0.0, 0.0],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lowest_level: bool = False,
    reset_used_patches_ids: bool = False,
    asset_configs: list[SceneEntityCfg] | None = None,
    reset_id: int = 0,
):
    """Reset the asset root state to a random position at the lowest level of the scene.
    This might be called multiple times to reset the root state of multiple assets.
    If assets must not be placed on the same position, the reset_used_patches_ids flag must be set to False
    for all but the first function call per reset."""

    # reset the used ids if required
    if reset_id == 0:
        # resample number of obstacles per env
        num_obs_range = env.cfg.data_container.num_obstacles_range
        env.cfg.data_container.num_obstacles = torch.randint(
            num_obs_range[0], num_obs_range[1] + 1, (len(env_ids),), dtype=torch.float
        ).to(env.device)

    # check if the asset should be removed from the scene
    spawn_lowest_terrain = reset_id < env.cfg.data_container.num_obstacles
    all_reset_env_ids = env_ids

    if reset_used_patches_ids:
        # reset the used patches ids, should be done only once per reset
        env.scene.terrain.terrain_used_flat_patches = {
            "lowest_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
            "init_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
            "not_lowest_pos": torch.zeros((len(env_ids), 1), dtype=torch.int64).to(env.device) - 1,
        }

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample random positions
    positions = torch.zeros((len(env_ids), 3), device=env.device)
    flat_patch_type = "lowest_pos" if lowest_level else "init_pos"
    positions[spawn_lowest_terrain] = _sample_terrain_pos(env, asset, env_ids, flat_patch_type)[spawn_lowest_terrain]
    positions[~spawn_lowest_terrain] = _sample_terrain_pos(env, asset, env_ids, "not_lowest_pos")[~spawn_lowest_terrain]
    positions += torch.tensor(offset, device=asset.device)

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def _sample_terrain_pos(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    flat_patch_type: str = "init_pos",
) -> torch.Tensor:
    """sample position that is on the terrain."""

    terrain: TerrainImporter = env.scene.terrain
    valid_positions: torch.Tensor = terrain.flat_patches.get(flat_patch_type)
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_uniform_on_terrain_aware' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    used_ids = terrain.terrain_used_flat_patches[flat_patch_type]
    ids = torch.zeros((len(env_ids),), dtype=torch.int64, device=env.device) - 1
    all_valid_per_env = torch.zeros((len(env_ids),), dtype=torch.bool, device=env.device)
    while not all_valid_per_env.all():
        ids[~all_valid_per_env] = torch.randint(
            0, valid_positions.shape[2], size=(int((~all_valid_per_env).sum()),), device=env.device
        )
        all_valid_per_env = torch.all(used_ids != ids.unsqueeze(1), dim=1)

    # add the used ids
    terrain.terrain_used_flat_patches[flat_patch_type] = torch.cat([used_ids, ids.unsqueeze(1)], dim=1)

    # get the positions
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]
    return positions
