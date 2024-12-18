# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

from . import hf_terrains


@configclass
class CellBorderCfg(HfTerrainBaseCfg):
    """Configuration for cell border wall."""

    function = hf_terrains.cell_border

    height: float = 2.0

    corner_witdh: float = 0.0


@configclass
class RandomPyramid(HfTerrainBaseCfg):
    """Configuration for cell border wall."""

    function = hf_terrains.random_pyramid

    wall_height: float = 2.0

    step_height: float = 0.5

    min_width: float = 0.5
    """minimal step with"""

    force_to_corner: bool = True
    """if true, all upper levels will be forced to the corner of the lower level.
    This will omit the min_width parameter"""

    origin_z: float = 0.0
