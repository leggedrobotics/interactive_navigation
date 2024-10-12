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
class MeshPyramidTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height: float = MISSING
    """The height of the steps (in m)."""
    step_width: tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    walls: bool = False
    """If True, each terrain is surrounded by walls. Defaults to False."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.
    

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """
