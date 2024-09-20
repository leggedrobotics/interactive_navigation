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
