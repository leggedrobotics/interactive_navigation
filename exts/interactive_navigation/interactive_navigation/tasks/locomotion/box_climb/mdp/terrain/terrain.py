# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg

from .mesh_terrains_cfg import StepTerrainCfg


height_range = (0.0, 1.0)
MESH_STEP_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    border_height=0.0,
    num_rows=8,  # difficulty levels
    num_cols=1,  # number of terrains per difficulty level
    horizontal_scale=0.5,
    vertical_scale=0.05,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # "step_1": StepTerrainCfg(
        #     proportion=1.0,
        #     step_height_range=height_range,
        #     step_width=(3.0, 3.0),
        # ),
        # "step_2": StepTerrainCfg(
        #     proportion=1.0,
        #     step_height_range=height_range,
        #     step_width=(2.0, 2.0),
        # ),
        # "step_3": StepTerrainCfg(
        #     proportion=0.5,
        #     step_height_range=height_range,
        #     step_width=(1.0, 1.0),
        # ),
        "step_4": StepTerrainCfg(
            proportion=0.5,
            step_height_range=height_range,
            step_width=(0.8, 0.6),
        ),
        # "nostep": StepTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.0),
        #     step_width=(8.0, 8.0),
        # ),
    },
)
