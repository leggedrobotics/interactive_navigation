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
    from interactive_navigation.tasks.locomotion.box_climb.mdp.terrain.mesh_terrains_cfg import (
        StepTerrainCfg,
    )

##


def reset_robot_around_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float],
    heading_range: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the root state around the step."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # position  :
    random_angles = torch.rand(len(env_ids), device=env.device) * torch.pi * 2
    random_radius = torch.rand(len(env_ids), device=env.device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    rand_x = random_radius * torch.cos(random_angles)
    rand_y = random_radius * torch.sin(random_angles)
    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + torch.cat([rand_x.unsqueeze(-1), rand_y.unsqueeze(-1), torch.zeros_like(rand_x.unsqueeze(-1))], dim=-1)
    )

    # heading:
    zero_heading = torch.atan2(rand_y, rand_x)
    random_heading = (
        torch.rand(len(env_ids), device=env.device) * (heading_range[1] - heading_range[0]) + heading_range[0]
    )
    heading = math_utils.wrap_to_pi(zero_heading + random_heading)

    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros(len(env_ids), device=env.device),
        torch.zeros(len(env_ids), device=env.device),
        heading,
    )

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
