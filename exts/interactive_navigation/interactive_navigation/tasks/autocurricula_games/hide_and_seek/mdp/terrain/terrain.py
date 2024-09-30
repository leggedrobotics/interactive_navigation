# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg

from .hf_terrains_cfg import CellBorderCfg, RandomPyramid

GAME_ARENA_BASE_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=0.0,
    border_height=0.0,
    curriculum=True,
    num_rows=16,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        # "rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=0.2,
        #     rail_thickness_range=(0.05, 0.2),
        #     rail_height_range=(0.05, 0.2),
        # ),
        "wall": CellBorderCfg(
            border_width=0,
            height=2.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(num_patches=25, patch_radius=0.4, max_height_diff=5.0),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=25, patch_radius=0.4, max_height_diff=0.5, z_range=(0.0, 0.5)
                ),
            },
        ),
    },
)

GAME_ARENA_RANDOM_FLOORS_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=0.0,
    border_height=0.0,
    curriculum=False,
    num_rows=16,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.25,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(1, 1),  # number of steps
    sub_terrains={
        # "rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=0.2,
        #     rail_thickness_range=(0.05, 0.2),
        #     rail_height_range=(0.05, 0.2),
        # ),
        "wall": RandomPyramid(
            border_width=0,
            wall_height=2.0,
            step_height=1,
            min_width=0.5,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(num_patches=128, patch_radius=0.4, max_height_diff=5.0),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=0.4, max_height_diff=0.5, z_range=(0.0, 0.5)
                ),
            },
        ),
    },
)
