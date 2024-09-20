# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from omni.isaac.lab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def cell_border(difficulty: float, cfg: hf_terrains_cfg.CellBorderCfg) -> np.ndarray:
    """Generate a cell border wall."""

    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))

    height = cfg.height / cfg.vertical_scale

    B = 1
    # -- border 1 pixels
    hf_raw[:B, :] = height
    hf_raw[-B:, :] = height
    hf_raw[:, :B] = height
    hf_raw[:, -B:] = height

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)
