# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import math
import trimesh
from typing import TYPE_CHECKING

from omni.isaac.lab.terrains.trimesh.utils import *  # noqa: F401, F403
from omni.isaac.lab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


import numpy as np
import trimesh
import random


def pyramid_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern where each level is placed at a random corner of the lower level.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Resolve the terrain configuration
    step_height = cfg.step_height  # + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # Compute number of steps
    log_skew = abs(cfg.step_width[1] - cfg.step_width[0]) * 2
    if log_skew != 0:
        step_width = cfg.step_width[0] + (math.log(difficulty * log_skew + 1) / math.log(log_skew + 1)) * (
            cfg.step_width[1] - cfg.step_width[0]
        )
    else:
        step_width = cfg.step_width[0] + difficulty * (cfg.step_width[1] - cfg.step_width[0])

    max_steps_x = (cfg.size[0] - 2 * cfg.border_width) // (2 * step_width)
    max_steps_y = (cfg.size[1] - 2 * cfg.border_width) // (2 * step_width)
    num_steps = int(min(max_steps_x, max_steps_y))

    # Initialize list of meshes
    meshes_list = []

    if cfg.walls:
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        # south wall
        center_south = [wall_thickness / 2, cfg.size[1] / 2, wall_height / 2]
        dims = [wall_thickness, cfg.size[1], wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_south))
        meshes_list.append(wall_box)
        # # north wall
        # center_north = [wall_thickness / 2, 0, wall_height / 2]
        # wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_north))
        # meshes_list.append(wall_box)
        # east wall
        center_east = [cfg.size[1] / 2, wall_thickness / 2, wall_height / 2]
        dims = [cfg.size[0], wall_thickness, wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_east))
        meshes_list.append(wall_box)
        # # west wall
        # center_west = [0, wall_thickness / 2, wall_height / 2]
        # wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_west))
        # meshes_list.append(wall_box)

        # Adjust the size of the terrain to account for the walls
        # cfg.size = (cfg.size[0] - cfg.wall_thickness, cfg.size[1] - cfg.wall_thickness)

    # Generate the border if needed
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        meshes_list += make_borders

    # Initialize variables for the base level
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    prev_box_size = terrain_size
    prev_box_pos = terrain_center
    prev_box_height = 0.0

    # Create the base level
    base_box_dims = (prev_box_size[0], prev_box_size[1], prev_box_height)
    base_box = trimesh.creation.box(base_box_dims, trimesh.transformations.translation_matrix(prev_box_pos))
    meshes_list.append(base_box)

    # Iterate through each level
    for k in range(1, num_steps + 1):
        # Reduce the size of the box for the next level
        box_size = (prev_box_size[0] - 2 * step_width, prev_box_size[1] - 2 * step_width)
        # Ensure the box size remains positive
        if box_size[0] <= cfg.platform_width or box_size[1] <= cfg.platform_width:
            break

        # Randomly select one of the four corners
        corners = [
            (-0.5, -0.5),  # Lower-left corner
            (-0.5, 0.5),  # Upper-left corner
            (0.5, -0.5),  # Lower-right corner
            (0.5, 0.5),  # Upper-right corner
        ]
        if cfg.type == "random":
            corner = random.choice(corners)
        elif cfg.type == "spiral":
            corner = corners[k % 4]
        else:
            corner = corners[0]

        # Calculate the position of the new box
        offset_x = corner[0] * (prev_box_size[0] - box_size[0])
        offset_y = corner[1] * (prev_box_size[1] - box_size[1])
        box_pos_x = prev_box_pos[0] + offset_x
        box_pos_y = prev_box_pos[1] + offset_y
        box_pos_z = prev_box_pos[2] + 0.5 * prev_box_height + 0.5 * step_height
        box_pos = (box_pos_x, box_pos_y, box_pos_z)

        # Create the new box
        box_dims = (box_size[0], box_size[1], step_height)
        new_box = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        meshes_list.append(new_box)

        # Update variables for the next iteration
        prev_box_size = box_size
        prev_box_pos = box_pos
        prev_box_height = step_height

    # Origin of the terrain
    # total_height = prev_box_pos[2] + 0.5 * prev_box_height
    origin = np.array([terrain_center[0], terrain_center[1], 0])

    return meshes_list, origin


def step_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a  single step at the -x,-y corner.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Resolve the terrain configuration

    # Initialize list of meshes
    meshes_list = []

    if cfg.walls:
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        # south wall
        center_south = [wall_thickness / 2, cfg.size[1] / 2, wall_height / 2]
        dims = [wall_thickness, cfg.size[1], wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_south))
        meshes_list.append(wall_box)
        # north wall
        center_east = [cfg.size[1] / 2, wall_thickness / 2, wall_height / 2]
        dims = [cfg.size[0], wall_thickness, wall_height]
        wall_box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center_east))
        meshes_list.append(wall_box)

    # Generate the border if needed
    if cfg.border_width > 0.0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.step_height / 2)
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, cfg.step_height, border_center)
        meshes_list += make_borders

    # Initialize variables for the base level
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)

    # Create the base level
    base_box_dims = (terrain_size[0], terrain_size[1], 0.0)
    base_box = trimesh.creation.box(base_box_dims, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(base_box)

    # Create the step
    step_box_dims = (cfg.step_width[0], cfg.step_width[1], cfg.step_height)
    step_center = [
        cfg.step_width[0] / 2,
        cfg.step_width[1] / 2,
        cfg.step_height / 2,
    ]
    step_box = trimesh.creation.box(step_box_dims, trimesh.transformations.translation_matrix(step_center))
    meshes_list.append(step_box)
    # Origin of the terrain
    # total_height = prev_box_pos[2] + 0.5 * prev_box_height
    origin = np.array([terrain_center[0], terrain_center[1], 0])

    return meshes_list, origin
