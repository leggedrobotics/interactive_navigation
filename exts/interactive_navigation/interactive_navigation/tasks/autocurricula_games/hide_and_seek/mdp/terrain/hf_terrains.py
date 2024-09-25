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

    hf_raw = np.zeros((width_pixels, length_pixels)).astype(bool)

    height = cfg.height / cfg.vertical_scale

    B = 1
    # -- border 1 pixels
    hf_raw[:B, :] = True
    hf_raw[-B:, :] = True
    hf_raw[:, :B] = True
    hf_raw[:, -B:] = True

    # cut corners
    if cfg.corner_witdh > 0:
        B = int(cfg.corner_witdh / cfg.horizontal_scale)
        # Top-left corner
        hf_raw[-B:, :B] |= np.tri(B, B, 0, dtype=bool)
        # Bottom-left corner
        hf_raw[:B, :B] |= np.tri(B, B, 0, dtype=bool)[::-1, :]
        # Top-right corner
        hf_raw[-B:, -B:] |= np.tri(B, B, 0, dtype=bool)[:, ::-1]
        # Bottom-right corner
        hf_raw[:B, -B:] |= np.tri(B, B, 0, dtype=bool)[::-1, ::-1]

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16) * height


@height_field_to_mesh
def random_pyramid(difficulty: float, cfg: hf_terrains_cfg.RandomPyramid) -> np.ndarray:
    """Generate a cell border wall."""

    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels)).astype(bool)

    height = cfg.wall_height / cfg.vertical_scale

    # - border 1 pixels
    B = 1
    # -- border 1 pixels
    hf_raw[:B, :] = True
    hf_raw[-B:, :] = True
    hf_raw[:, :B] = True
    hf_raw[:, -B:] = True

    hf_raw = hf_raw.astype(float) * height

    # - pyramid

    # we generate a random pyramid by creating square levels of increasing width
    # with random xy offsets
    N_levels = cfg.N_steps
    avg_level_width = width_pixels / (2 * N_levels + 1)
    step_height = int(cfg.step_height / cfg.vertical_scale)

    # # random overall
    # for level_i in range(N_levels):
    #     level_width = int((level_i * 2 + 1) * avg_level_width)
    #     start_x = np.random.randint(0, width_pixels - level_width)
    #     start_y = np.random.randint(0, length_pixels - level_width)
    #     hf_raw[start_x : start_x + level_width, start_y : start_y + level_width] += step_height

    # random but smaller in bigger levels
    min_step_width_pixels = int(cfg.min_width / cfg.horizontal_scale)
    big_start_x = big_start_y = 0
    big_width = width_pixels
    for level_i in reversed(range(N_levels)):
        level_width = int((level_i * 2 + 1) * avg_level_width)
        # start_x = big_start_x + int((big_width - level_width) / 2)
        start_x = big_start_x + np.random.randint(
            min_step_width_pixels, big_width - level_width - min_step_width_pixels
        )
        # start_y = big_start_y + int((big_width - level_width) / 2)
        start_y = big_start_y + np.random.randint(
            min_step_width_pixels, big_width - level_width - min_step_width_pixels
        )
        hf_raw[start_x : start_x + level_width, start_y : start_y + level_width] += step_height
        big_start_x = start_x
        big_start_y = start_y
        big_width = level_width

    # round off the heights to the nearest vertical step

    if False:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("TkAgg")
        plt.imshow(hf_raw)
        plt.show()

    return np.rint(hf_raw).astype(np.int16)
