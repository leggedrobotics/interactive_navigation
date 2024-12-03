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
from omni.isaac.lab.managers import CurriculumTermCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase

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


class anneal_reward_weight(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.initial_weights = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_names: list[str],
        ratio: float,
        start_step: int,
        num_steps: int,
    ):
        """Curriculum that modifies a reward weight a given number of steps.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            term_names: The name of the reward term.
            ratio: The final reward will have weight initial weight * ratio.
            start_step: The number of steps after which the change should be applied.
            num_steps: The number of over which the weight should be annealed.
        """

        if self.initial_weights is None:
            self.initial_weights = {
                term_name: env.reward_manager.get_term_cfg(term_name).weight for term_name in term_names
            }

        if env.common_step_counter > start_step:
            # interpolate ratios from 1 to ratio
            current_ratio = 1 + (ratio - 1) * (env.common_step_counter - start_step) / num_steps
            if env.common_step_counter > start_step + num_steps:
                current_ratio = ratio

            for term_name in term_names:
                # obtain term settings
                term_cfg = env.reward_manager.get_term_cfg(term_name)
                # update term settings
                term_cfg.weight = self.initial_weights[term_name] * current_ratio
                env.reward_manager.set_term_cfg(term_name, term_cfg)

            return current_ratio
        return 1.0
