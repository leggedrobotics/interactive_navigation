from __future__ import annotations

import numpy as np
import math
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
    from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.terrain.mesh_terrains_cfg import (
        StepTerrainCfg,
    )

##
# reset terrain aware
##


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


##
# reset joints
##
def reset_id_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset specific robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # find joint ids:
    joint_ids, _ = asset.find_joints(joint_names)
    joint_ids = tuple(joint_ids)

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids[:, None], joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids[:, None], joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


##
# reset near step
##


def reset_near_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    max_dist: float = 4.0,
    level: int = 0,
    min_offset: float = 0.5,
    random_dist: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot to a position near the step."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample random positions around the step
    if random_dist:
        # box_from_step_dist and box_from_step_dist are hacked in to the env in the curriculum
        dist_tensor = (
            torch.rand_like(env.box_from_step_dist) * torch.clip(env.box_from_step_dist - min_offset, min=0)
            + min_offset
        )
    else:
        dist_tensor = torch.clip(env.box_from_step_dist, 0.2, max_dist)
    positions = _sample_pos_near_step(
        dist=dist_tensor, min_offset=0.5, terrain=env.scene.terrain, env_ids=env_ids, level=level
    )

    # z offset:
    positions[:, 2] += 0.5

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1).float(), env_ids=env_ids)


def reset_boxes_and_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    boxes_sorted: list[SceneEntityCfg],
    pose_range: dict[str, tuple[float, float]],
    random_dist: bool = False,
    min_dist: float = 2.0,
    robot_z_offset: float = 0.5,
):
    """Reset function for env with one big step and multiple boxes.
    given the "dist" param in the env, set in the curriculum, the boxes will
    be spawned ordered from the step, such that the first box is the closest to the step,
    the second is close to the first, etc, until the last is close to the robot.

    Note: the distance is from origin to origin, so the size of the boxes is not considered here.
    Note: the terrain for this function should contain one step.The step is in the -x,-y corner of the terrain.
    Additionally, the terrain should be constant, i.e., it should not have randomness or difficulty

    Terrain:
    #: step
    x: box
    r: robot
        +---------------------+
        |#####                |
        |#####  x             |
        |#####    x  x        |
        |              x   r  |
        |                     |
        +---------------------+
    """
    num_entities = len(boxes_sorted) + 1

    # find step corner:
    terrain = env.scene.terrain.cfg.terrain_generator
    assert len(terrain.sub_terrains) == 1, "The terrain should contain only one sub terrain"
    step_terrain: StepTerrainCfg = list(terrain.sub_terrains.values())[0]

    terrain_size = terrain.size
    step_size = step_terrain.step_width

    min_x = -terrain_size[0] / 2 + step_size[0]
    min_y = -terrain_size[1] / 2 + step_size[1]

    # terrain origins positions
    terrain_origins = env.scene.terrain.env_origins[env_ids]

    # sample distances if random dist
    max_dist_per_env: torch.Tensor = env.dist[env_ids]
    if random_dist:
        # random distances within the bounds
        distances = (
            torch.rand(len(env_ids), num_entities, device=env.device) * (max_dist_per_env.unsqueeze(1) - min_dist)
            + min_dist
        )
    else:
        # always use the upper bounds
        distances = max_dist_per_env.unsqueeze(1).repeat(1, num_entities)

    chain_positions = sample_chain_positions(
        num_envs=len(env_ids),
        N=num_entities,
        r_min=min_dist,
        distances=distances,
        terrain_size_x=terrain_size[0],
        terrain_size_y=terrain_size[1],
        x_step=min_x,
        y_step=min_y,
    )

    assert not chain_positions.isnan().any(), "NaN values in chain_positions"

    terrain_origins_2d = terrain_origins[:, :2].unsqueeze(1)
    chain_positions += terrain_origins_2d

    # set boxes
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    for i, box_cfg in enumerate(boxes_sorted):
        box = env.scene[box_cfg.name]

        # sample orientation
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)
        orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

        # get z offset
        z_offset = torch.zeros_like(env_ids) + box.cfg.spawn.size[2] / 2 + 0.01

        # set box pose
        box_poses = torch.cat([chain_positions[:, i + 1], z_offset.unsqueeze(1), orientations], dim=-1).float()
        box.write_root_pose_to_sim(box_poses.float(), env_ids=env_ids)

    # set robot
    robot = env.scene[robot_cfg.name]
    orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device).repeat(len(env_ids), 1)
    z_offset = torch.zeros_like(env_ids) + robot_z_offset
    robot_pose = torch.cat([chain_positions[:, -1], z_offset.unsqueeze(1), orientations], dim=-1).float()
    robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)


""" 

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Rectangle

    matplotlib.use("TkAgg")

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(env_ids)):
        entity_points = chain_positions[i].cpu().numpy()

        ax.plot(entity_points[:, 0], entity_points[:, 1], "o-")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-terrain_size[0] / 2, terrain_size[0] / 2)
        ax.set_ylim(-terrain_size[1] / 2, terrain_size[1] / 2)

        # drwa step
        ax.add_patch(
            Rectangle(
                (min_x - step_size[0], min_y - step_size[1]),
                step_size[0],
                step_size[1],
                edgecolor="black",
                linewidth=2,
            )
        )
    plt.show()


"""


def sample_chain_positions(
    num_envs: int,
    N: int,
    r_min: float,
    distances: torch.Tensor,
    terrain_size_x: float,
    terrain_size_y: float,
    x_step: float,
    y_step: float,
    max_attempts_per_step: int = 50,
    max_total_attempts: int = 1000,
) -> torch.Tensor:
    """
    Samples N positions in 2D space for multiple environments in parallel,
    considering terrain constraints and minimum distance requirements.
    Ensures no failed positions by using backtracking and restarting when necessary.
    Adjusted to sample the angle randomly for the first point if the first distance
    is greater than 2 * r_min.
    """
    device = distances.device
    dtype = distances.dtype

    pi = math.pi

    # Terrain boundaries
    x_min = -terrain_size_x / 2
    x_max = terrain_size_x / 2
    y_min = -terrain_size_y / 2
    y_max = terrain_size_y / 2

    # Adjusted terrain boundaries considering r_min
    x_min_valid = x_min + r_min
    x_max_valid = x_max - r_min
    y_min_valid = y_min + r_min
    y_max_valid = y_max - r_min

    # Adjusted step area considering r_min
    x_step_edge = x_step
    y_step_edge = y_step

    # Initialize positions tensor
    positions = torch.zeros(num_envs, N + 1, 2, device=device, dtype=dtype)

    # Function to initialize starting positions and first point
    def initialize_starting_positions(env_indices):
        num_envs_init = env_indices.shape[0]
        # Randomly select vertical or horizontal edge for starting position
        edge_choice = torch.randint(0, 2, (num_envs_init,), device=device)  # 0: vertical, 1: horizontal

        # Starting positions for vertical edge (x = x_step + r_min)
        x_start_vertical = x_step_edge * torch.ones(num_envs_init, device=device, dtype=dtype)
        y_start_vertical = (
            torch.rand(num_envs_init, device=device, dtype=dtype) * (y_step_edge - y_min_valid) + y_min_valid
        )

        # Starting positions for horizontal edge (y = y_step + r_min)
        x_start_horizontal = (
            torch.rand(num_envs_init, device=device, dtype=dtype) * (x_step_edge - x_min_valid) + x_min_valid
        )
        y_start_horizontal = y_step_edge * torch.ones(num_envs_init, device=device, dtype=dtype)

        # Combine starting positions based on edge choice
        x_start = torch.where(edge_choice == 0, x_start_vertical, x_start_horizontal)
        y_start = torch.where(edge_choice == 0, y_start_vertical, y_start_horizontal)

        positions[env_indices, 0, 0] = x_start
        positions[env_indices, 0, 1] = y_start

        # First point is determined based on distances_first and r_min
        distances_first = distances[env_indices, 0]  # Shape: [num_envs_init]

        # Create a mask for distances_first > 2 * r_min
        mask_random = distances_first > (2 * r_min)

        # For envs where mask_random is True, sample random angles
        num_random = mask_random.sum().item()
        if num_random > 0:
            random_indices = mask_random.nonzero(as_tuple=False).squeeze(-1)
            random_distances = distances_first[random_indices]

            # Sample random angles
            random_angles = torch.rand(num_random, device=device, dtype=dtype) * 2 * pi
            delta_x_random = torch.cos(random_angles) * random_distances
            delta_y_random = torch.sin(random_angles) * random_distances

            positions[env_indices[random_indices], 1, 0] = positions[env_indices[random_indices], 0, 0] + delta_x_random
            positions[env_indices[random_indices], 1, 1] = positions[env_indices[random_indices], 0, 1] + delta_y_random

        # For envs where mask_random is False, move along x or y depending on edge_choice
        mask_direct = ~mask_random
        num_direct = mask_direct.sum().item()
        if num_direct > 0:
            direct_indices = mask_direct.nonzero(as_tuple=False).squeeze(-1)
            direct_distances = distances_first[direct_indices]

            delta_x_direct = torch.where(
                edge_choice[direct_indices] == 0, direct_distances, torch.zeros(num_direct, device=device, dtype=dtype)
            )
            delta_y_direct = torch.where(
                edge_choice[direct_indices] == 1, direct_distances, torch.zeros(num_direct, device=device, dtype=dtype)
            )

            positions[env_indices[direct_indices], 1, 0] = positions[env_indices[direct_indices], 0, 0] + delta_x_direct
            positions[env_indices[direct_indices], 1, 1] = positions[env_indices[direct_indices], 0, 1] + delta_y_direct

        # Check validity of the first point and handle invalid positions
        # Combine indices and positions
        all_indices = (
            torch.cat([random_indices, direct_indices])
            if num_random > 0 and num_direct > 0
            else (random_indices if num_random > 0 else direct_indices)
        )
        if all_indices.numel() > 0:
            x_first = positions[env_indices[all_indices], 1, 0]
            y_first = positions[env_indices[all_indices], 1, 1]

            within_terrain = (
                (x_first >= x_min_valid)
                & (x_first <= x_max_valid)
                & (y_first >= y_min_valid)
                & (y_first <= y_max_valid)
            )
            outside_step = (x_first > x_step_edge) | (y_first > y_step_edge)

            distance_to_start = torch.norm(
                positions[env_indices[all_indices], 1, :] - positions[env_indices[all_indices], 0, :], dim=1
            )
            distance_valid = distance_to_start >= r_min

            valid = within_terrain & outside_step & distance_valid

            invalid_indices = all_indices[~valid]
            attempts = 0
            max_attempts = 10
            while invalid_indices.numel() > 0 and attempts < max_attempts:
                # Resample for invalid indices
                # For these invalid indices, determine if they were in random_indices or direct_indices
                is_random = distances_first[invalid_indices] > (2 * r_min)
                random_invalid_indices = invalid_indices[is_random]
                direct_invalid_indices = invalid_indices[~is_random]

                # Resample random angles for random_invalid_indices
                if random_invalid_indices.numel() > 0:
                    num_random_invalid = random_invalid_indices.numel()
                    random_angles = torch.rand(num_random_invalid, device=device, dtype=dtype) * 2 * pi
                    random_distances = distances_first[random_invalid_indices]

                    delta_x_random = torch.cos(random_angles) * random_distances
                    delta_y_random = torch.sin(random_angles) * random_distances

                    positions[env_indices[random_invalid_indices], 1, 0] = (
                        positions[env_indices[random_invalid_indices], 0, 0] + delta_x_random
                    )
                    positions[env_indices[random_invalid_indices], 1, 1] = (
                        positions[env_indices[random_invalid_indices], 0, 1] + delta_y_random
                    )

                # For direct_invalid_indices, we can keep the same movement along x or y
                # but since it's invalid, we need to handle it. For simplicity, we can adjust distances to be r_min
                if direct_invalid_indices.numel() > 0:
                    direct_distances = torch.full_like(distances_first[direct_invalid_indices], r_min, device=device)
                    edge_choice_invalid = edge_choice[direct_invalid_indices]

                    delta_x_direct = torch.where(
                        edge_choice_invalid == 0,
                        direct_distances,
                        torch.zeros(direct_invalid_indices.numel(), device=device, dtype=dtype),
                    )
                    delta_y_direct = torch.where(
                        edge_choice_invalid == 1,
                        direct_distances,
                        torch.zeros(direct_invalid_indices.numel(), device=device, dtype=dtype),
                    )

                    positions[env_indices[direct_invalid_indices], 1, 0] = (
                        positions[env_indices[direct_invalid_indices], 0, 0] + delta_x_direct
                    )
                    positions[env_indices[direct_invalid_indices], 1, 1] = (
                        positions[env_indices[direct_invalid_indices], 0, 1] + delta_y_direct
                    )

                # Recalculate validity
                x_first = positions[env_indices[invalid_indices], 1, 0]
                y_first = positions[env_indices[invalid_indices], 1, 1]

                within_terrain = (
                    (x_first >= x_min_valid)
                    & (x_first <= x_max_valid)
                    & (y_first >= y_min_valid)
                    & (y_first <= y_max_valid)
                )
                outside_step = (x_first > x_step_edge) | (y_first > y_step_edge)

                distance_to_start = torch.norm(
                    positions[env_indices[invalid_indices], 1, :] - positions[env_indices[invalid_indices], 0, :], dim=1
                )
                distance_valid = distance_to_start >= r_min

                valid = within_terrain & outside_step & distance_valid

                invalid_indices = invalid_indices[~valid]
                attempts += 1

            # If after max_attempts there are still invalid positions, restart those environments
            if invalid_indices.numel() > 0:
                restart_env_indices = env_indices[invalid_indices]
                # Restart sampling for these environments
                initialize_starting_positions(restart_env_indices)

    # Initialize step indices and attempt counters
    step_indices = torch.full((num_envs,), 2, device=device, dtype=torch.long)  # Start from the second point
    attempt_counters = torch.zeros(num_envs, device=device, dtype=torch.long)
    total_attempt_counters = torch.zeros(num_envs, device=device, dtype=torch.long)
    active_envs = torch.ones(num_envs, dtype=torch.bool, device=device)

    # Initialize starting positions for all environments
    env_indices = torch.arange(num_envs, device=device)
    initialize_starting_positions(env_indices)

    # Main sampling loop
    while active_envs.any():
        active_env_indices = active_envs.nonzero(as_tuple=False).squeeze(-1)
        num_active_envs = active_env_indices.numel()
        if num_active_envs == 0:
            break

        current_steps = step_indices[active_env_indices]

        # Identify environments that have completed sampling
        finished_envs = current_steps > N
        if finished_envs.any():
            finished_indices = active_env_indices[finished_envs.nonzero(as_tuple=False).squeeze(-1)]
            active_envs[finished_indices] = False
            continue

        # Prepare sampling for current active environments
        K = 30  # Number of candidate angles per attempt
        prev_positions = positions[active_env_indices, current_steps - 1, :]  # [num_active_envs, 2]
        step_distances = distances[active_env_indices, current_steps - 1]  # [num_active_envs]

        # Sample candidate angles
        angles = torch.rand(num_active_envs * K, device=device, dtype=dtype) * 2 * pi
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        distances_expanded = step_distances.repeat_interleave(K)
        delta_x = cos_angles * distances_expanded
        delta_y = sin_angles * distances_expanded

        prev_positions_expanded = prev_positions.repeat_interleave(K, dim=0)
        candidate_positions = prev_positions_expanded + torch.stack([delta_x, delta_y], dim=1)

        # Check constraints
        x = candidate_positions[:, 0]
        y = candidate_positions[:, 1]

        # Valid positions within terrain boundaries
        within_terrain = (x >= x_min_valid) & (x <= x_max_valid) & (y >= y_min_valid) & (y <= y_max_valid)

        # Outside expanded step area
        outside_step = (x > x_step_edge) | (y > y_step_edge)

        # Initialize validity masks
        valid_positions_mask = within_terrain & outside_step
        valid_candidate_found = torch.zeros(num_active_envs, dtype=torch.bool, device=device)
        selected_candidate_positions = torch.zeros(num_active_envs, 2, device=device, dtype=dtype)

        # Check distances to previous points
        for idx in range(num_active_envs):
            env_idx = active_env_indices[idx]
            current_step = current_steps[idx].item()
            num_prev_positions = current_step  # positions from 0 to current_step - 1

            candidate_idx_start = idx * K
            candidate_idx_end = candidate_idx_start + K

            valid_candidates_env = valid_positions_mask[candidate_idx_start:candidate_idx_end]
            if valid_candidates_env.any():
                # Get valid candidate positions
                candidate_positions_env = candidate_positions[candidate_idx_start:candidate_idx_end][
                    valid_candidates_env
                ]
                num_candidates_env = candidate_positions_env.shape[0]
                # Compute distances to previous positions
                prev_positions_env = positions[env_idx, :current_step, :]  # [current_step, 2]
                deltas = candidate_positions_env.unsqueeze(1) - prev_positions_env.unsqueeze(0)
                distances_to_prev = torch.norm(deltas, dim=2)
                min_distances = distances_to_prev.min(dim=1)[0]
                distance_valid = min_distances >= r_min
                if distance_valid.any():
                    # Select first valid candidate
                    first_valid_idx = distance_valid.nonzero(as_tuple=False)[0].item()
                    selected_candidate_positions[idx, :] = candidate_positions_env[first_valid_idx, :]
                    valid_candidate_found[idx] = True

        # Update positions and counters
        envs_with_valid_candidates = valid_candidate_found.nonzero(as_tuple=False).squeeze(-1)
        if envs_with_valid_candidates.numel() > 0:
            env_indices = active_env_indices[envs_with_valid_candidates]
            positions[env_indices, current_steps[envs_with_valid_candidates], :] = selected_candidate_positions[
                envs_with_valid_candidates, :
            ]
            # Increment step indices
            step_indices[env_indices] += 1
            # Reset attempt counters
            attempt_counters[env_indices] = 0
            total_attempt_counters[env_indices] += 1

        # Handle environments without valid candidates
        envs_without_valid_candidates = (~valid_candidate_found).nonzero(as_tuple=False).squeeze(-1)
        if envs_without_valid_candidates.numel() > 0:
            env_indices = active_env_indices[envs_without_valid_candidates]
            attempt_counters[env_indices] += 1
            total_attempt_counters[env_indices] += 1

            # Backtracking if attempts exceed max_attempts_per_step
            over_attempt_limit = attempt_counters[env_indices] >= max_attempts_per_step
            if over_attempt_limit.any():
                over_limit_env_indices = env_indices[over_attempt_limit.nonzero(as_tuple=False).squeeze(-1)]
                # Backtrack
                step_indices[over_limit_env_indices] -= 1
                attempt_counters[over_limit_env_indices] = 0
                # If step_indices <= 1, restart sampling
                need_restart = step_indices[over_limit_env_indices] <= 1
                if need_restart.any():
                    restart_env_indices = over_limit_env_indices[need_restart.nonzero(as_tuple=False).squeeze(-1)]
                    # Restart sampling for these environments
                    initialize_starting_positions(restart_env_indices)
                    step_indices[restart_env_indices] = 2
                    attempt_counters[restart_env_indices] = 0
                    total_attempt_counters[restart_env_indices] = 0
                # For other environments, continue backtracking
                else:
                    attempt_counters[over_limit_env_indices] = 0

            # Check total attempts to prevent infinite loops
            over_total_attempt_limit = total_attempt_counters[env_indices] >= max_total_attempts
            if over_total_attempt_limit.any():
                fail_env_indices = env_indices[over_total_attempt_limit.nonzero(as_tuple=False).squeeze(-1)]
                # Restart sampling for these environments
                initialize_starting_positions(fail_env_indices)
                step_indices[fail_env_indices] = 2
                attempt_counters[fail_env_indices] = 0
                total_attempt_counters[fail_env_indices] = 0

    return positions


##
# Old stuff
##
def reset_box_near_step_and_robot_near_box(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    box_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg,
    min_robot_dist=2.0,
    min_box_dist=0.5,
    random_dist: bool = True,
):
    robot: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    box: RigidObject = env.scene[box_asset_cfg.name]

    # get positions
    if random_dist:
        # env.box_from_step_dist and env.robot_from_box_dist are upper bounds
        dist_box_to_step = (
            torch.rand_like(env.box_from_step_dist) * torch.clip(env.box_from_step_dist - min_box_dist, min=0)
            + min_box_dist
        )
        dist_robot_to_box = (
            torch.rand_like(env.robot_from_box_dist) * torch.clip(env.robot_from_box_dist - min_robot_dist, min=0)
            + min_robot_dist
        )
    else:
        # always use the upper bounds
        dist_box_to_step = env.box_from_step_dist
        dist_robot_to_box = env.robot_from_box_dist

    box_pos = _sample_pos_near_step(dist=dist_box_to_step, min_offset=0.5, terrain=env.scene.terrain, env_ids=env_ids)
    robot_pos = _sample_pos_near_box(
        dist=dist_robot_to_box, box_positions=box_pos, terrain=env.scene.terrain, env_ids=env_ids
    )

    # place box
    box_pos[:, 2] += 0.5
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=robot.device)
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])
    box.write_root_pose_to_sim(torch.cat([box_pos, orientations], dim=-1).float(), env_ids=env_ids)

    # place robot
    robot_pos = _sample_pos_near_box(
        dist=dist_robot_to_box, box_positions=box_pos, terrain=env.scene.terrain, env_ids=env_ids
    )
    orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device).repeat(len(env_ids), 1)
    robot.write_root_pose_to_sim(torch.cat([robot_pos, orientations], dim=-1).float(), env_ids=env_ids)


def _sample_pos_near_step(
    dist: torch.Tensor, min_offset: float, terrain: TerrainImporter, env_ids: torch.Tensor, level: int = 0
) -> torch.Tensor:
    """This function samples the a point near the step.
    Assumes the pyramid is skewed to the -x, -y direction."""

    # wp_terrain_mesh = terrain.warp_meshes["terrain"]
    size = terrain.cfg.terrain_generator.size  # (width, height)
    wall_height = terrain.cfg.terrain_generator.border_height
    # we sample a point for each env_origin
    mesh_points = torch.tensor(terrain.meshes["terrain"].vertices).to(terrain.device)
    random_points = []

    for env_id in env_ids:
        # find points in terrain
        env_origin = terrain.env_origins[env_id]
        env_dist = dist[env_id]
        diff = torch.abs(env_origin - mesh_points)
        valid_points = (
            (diff[:, 0] < size[0] / 2 - 0.05)
            & (diff[:, 1] < size[1] / 2 - 0.05)
            & (mesh_points[:, 2] < wall_height)
            & (mesh_points[:, 2] > 0)
        )
        points = mesh_points[valid_points]
        # we find the height of the desired level
        # if level is bigger than the number of unique heights, we choose a random one
        unique_heights = points[:, 2].unique().sort()[0]
        if len(unique_heights) == 0:
            # sample random point
            random_offset = ((torch.rand(3) - 0.5) * torch.tensor([size[0], size[1], 0])).to(terrain.device)
            random_points.append(env_origin + random_offset)
            continue

        if level >= len(unique_heights):
            _level = int(torch.randint(0, len(unique_heights), (1,)).item())
        else:
            _level = level

        level_height = unique_heights[_level]
        min_step_points = points[points[:, 2] == level_height]
        lowest_corner = min_step_points.max(dim=0)[0]
        lowest_corner[2] -= unique_heights[0]

        # random shift
        left_right_space = (lowest_corner - env_origin)[0] + size[0] / 2 - min_offset
        random_shift = torch.rand(1).to(terrain.device) * left_right_space
        in_x = torch.rand(1).to(terrain.device) < 0.5

        corner_offset = torch.zeros(3).to(terrain.device)
        corner_offset[0] = -random_shift if in_x else min_offset + env_dist
        corner_offset[1] = -random_shift if not in_x else min_offset + env_dist

        random_points.append(lowest_corner + corner_offset)

    return torch.stack(random_points)


def _sample_pos_near_box(
    dist: torch.Tensor, box_positions: torch.Tensor, terrain: TerrainImporter, env_ids: torch.Tensor
) -> torch.Tensor:
    """Sample a point near the box, such that its on the same terrain level as the box.
    We use raycasting."""

    wp_terrain_mesh = terrain.warp_meshes["terrain"]

    # create a circular ray pattern:
    N_rays = 200
    angles = torch.linspace(0, 2 * torch.pi, N_rays + 1)[:-1].to(terrain.device)
    ray_starts = torch.stack([torch.cos(angles), torch.sin(angles), torch.ones_like(angles) + 100]).T

    # ray_starts_envs = box_positions.unsqueeze(1) + ray_starts.unsqueeze(0)

    ray_directions = torch.stack([torch.zeros(N_rays), torch.zeros(N_rays), -torch.ones(N_rays)]).T.to(
        terrain.device
    )  # .repeat(env_ids.shape[0], 1, 1)

    robot_positions = []
    for i, env_id in enumerate(env_ids):
        # scale radius
        sub_ray_starts = ray_starts.clone()
        sub_ray_starts[:, :2] *= dist[env_id]
        # shift to box position
        sub_ray_starts += box_positions[i]
        ray_hits = raycast_mesh(
            sub_ray_starts.unsqueeze(0).float().contiguous(),
            ray_directions.unsqueeze(0).float().contiguous(),
            wp_terrain_mesh.id,
        )[0].squeeze(0)

        # find points on lowest level
        valid = ray_hits[:, 2] < 0.1
        # we need to not sample points directly at an edge
        # we do this by trimming left and right from sequences of valid points
        # we remove as many points as the width of the box
        robot_width = 0.5
        width_per_point = torch.pi * dist[env_id] / N_rays
        N_trim = int(robot_width / width_per_point)
        valid = _trim_sequences(valid, N_trim)

        # random sample
        valid_points = ray_hits[valid]
        if len(valid_points) != 0:
            random_point = valid_points[torch.randint(0, valid_points.shape[0], (1,))].squeeze()
        else:
            random_point = box_positions[i] + torch.tensor([0.0, 0.0, 0.5]).to(terrain.device)
        robot_positions.append(random_point)
    return torch.stack(robot_positions)


def _trim_sequences(tensor: torch.Tensor, N: int) -> torch.Tensor:
    # Ensure the tensor is a 1D boolean tensor
    if tensor.dim() != 1 or tensor.dtype != torch.bool:
        raise ValueError("Input must be a 1D boolean tensor.")

    length = len(tensor)

    # Convert boolean tensor to integer tensor for calculations
    int_tensor = tensor.int()

    # Pad the tensor at both ends with zeros to handle edge cases
    padded = torch.nn.functional.pad(int_tensor, (1, 1), value=0)

    # Compute the differences to find the start and end indices of sequences
    diffs = padded[1:] - padded[:-1]

    # Start indices where the value changes from 0 to 1
    starts = torch.where(diffs == 1)[0]

    # End indices where the value changes from 1 to 0
    ends = torch.where(diffs == -1)[0] - 1  # Subtract 1 to make 'ends' inclusive

    # Adjust starts and ends by trimming N ones from both sides
    adjusted_starts = starts + N
    adjusted_ends = ends - N

    # Clamp the adjusted indices to be within the valid range
    adjusted_starts = adjusted_starts.clamp(0, length - 1)
    adjusted_ends = adjusted_ends.clamp(0, length - 1)

    # Initialize the output tensor with zeros (False)
    output = torch.zeros_like(tensor, dtype=torch.bool)

    # Efficiently set the trimmed sequences to True
    for start, end in zip(adjusted_starts, adjusted_ends):
        if start <= end:
            output[start : end + 1] = True  # end + 1 because the end index is inclusive

    return output
