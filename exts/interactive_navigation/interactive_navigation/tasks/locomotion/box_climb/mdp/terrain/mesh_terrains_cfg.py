# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from . import mesh_terrains as mesh_terrains
import omni.isaac.lab.terrains.trimesh.utils as mesh_utils_terrains
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


@configclass
class StepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.step_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The height of the steps (in m)."""
    step_width: tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    step_offset: float = 0.0
    """The offset of the steps (in m). Defaults to 0.0."""
    walls: bool = False
    """If True, each terrain is surrounded by walls. Defaults to False."""
    wall_height: float = 2.0
    """The height of the walls (in m). Defaults to 2.0."""
    wall_thickness: float = 0.1
