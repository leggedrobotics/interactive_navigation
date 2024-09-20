# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_cfg import CellBorderCfg

GAME_ARENA_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=0.0,
    border_height=0.0,
    num_rows=4,
    num_cols=4,
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
            height=1.0,
        ),
    },
)
"""Rough terrains configuration."""
