"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    move_down = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    move_up = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float()).item()


def num_boxes_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the number of boxes in the scene.

    This term is used to increase the number of boxes in the scene when the robot walks far enough and decrease the
    number of boxes when the robot walks less than half of the distance required by the commanded velocity.

    Returns:
        The mean number of boxes for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)

    # return the mean terrain level

    num_range = env.cfg.data_container.num_obstacles_range
    return (num_range[0] + num_range[1]) / 2


def box_from_step_dist_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]):
    env.box_from_step_dist_curriculum = 1

    return env.box_from_step_dist_curriculum


def robot_from_box_dist_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]):
    env.robot_from_box_dist_curriculum = 2

    return env.robot_from_box_dist_curriculum
