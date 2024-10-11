# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg

from .hf_terrains_cfg import CellBorderCfg, RandomPyramid
from .mesh_terrains_cfg import MeshPyramidTerrainCfg

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
    border_width=20.0,
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
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=0.4, max_height_diff=5.0, z_range=(0.75, 5.0)
                ),
            },
        ),
    },
)


MESH_PYRAMID_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=1.0,
    border_height=10.0,
    num_rows=16,  # difficulty levels
    num_cols=16,  # number of terrains per difficulty level
    horizontal_scale=0.5,
    vertical_scale=0.05,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0, 0.75),
    sub_terrains={
        "pyramid_stairs": MeshPyramidTerrainCfg(
            proportion=1.0,
            step_height_range=1.0,
            step_width=(5.0, 1.5),
            platform_width=3.0,
            border_width=0.0,
            holes=False,
            walls=True,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=1.5,
                    max_height_diff=5.0,
                ),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128, patch_radius=1.0, max_height_diff=0.5, z_range=(0.0, 0.25)
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.3,
                    max_height_diff=50.0,
                    z_range=(0.75, 50.0),
                ),
            },
        )
    },
)


MESH_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    border_height=0.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=1.0,
    vertical_scale=1.0,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "pyramid_stairs": MeshPyramidTerrainCfg(
            proportion=1.0,
            step_height_range=0.0,
            step_width=(100.0, 100.5),
            platform_width=3.0,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=1.5,
                    max_height_diff=5.0,
                ),
                "lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.5,
                    max_height_diff=0.5,
                ),
                "not_lowest_pos": FlatPatchSamplingCfg(
                    num_patches=128,
                    patch_radius=0.3,
                    max_height_diff=50.0,
                ),
            },
        )
    },
)
