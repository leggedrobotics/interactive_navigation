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


class DistanceCurriculum:
    # TODO implement the curriculum for the distance between the robot and the box
    # increase distance on termination, when the reward new height is reached

    def __init__(
        self,
        min_box_step_dist: float = 0.2,
        min_robot_box_dist: float = 2.0,
        max_box_step_dist: float = 4.0,
        max_robot_box_dist: float = 10.0,
        box_step_dist_increment: float = 0.1,
        robot_box_dist_increment: float = 0.1,
        goal_termination_name: str = "goal_reached",
    ):

        self.goal_termination_name = goal_termination_name
        self.min_box_step_dist = min_box_step_dist
        self.init_robot_box_dist = min_robot_box_dist
        self.max_box_step_dist = max_box_step_dist
        self.max_robot_box_dist = max_robot_box_dist
        self.box_step_dist_increment = box_step_dist_increment
        self.robot_box_dist_increment = robot_box_dist_increment
        # buffers
        self.box_from_step_dist: torch.Tensor = None
        self.robot_from_box_dist: torch.Tensor = None

    def _update(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]):

        if self.box_from_step_dist is None:
            self.box_from_step_dist = torch.ones(env.num_envs, device=env.device) * self.min_box_step_dist
        if self.robot_from_box_dist is None:
            self.robot_from_box_dist = torch.ones(env.num_envs, device=env.device) * self.init_robot_box_dist

        terminated_at_goal = env.termination_manager._term_dones[self.goal_termination_name]
        terminated = env.termination_manager.dones
        terminated_not_at_goal = terminated & ~terminated_at_goal
        if terminated_at_goal.any():
            # increase distance if goal was reached
            self.box_from_step_dist[env_ids] += self.box_step_dist_increment
            self.robot_from_box_dist[env_ids] += self.robot_box_dist_increment
        elif terminated_not_at_goal.any():
            # decrease distance if goal was not reached
            self.box_from_step_dist[env_ids] -= self.box_step_dist_increment
            self.robot_from_box_dist[env_ids] -= self.robot_box_dist_increment

        # clamp the values
        self.box_from_step_dist[env_ids] = torch.clamp(
            self.box_from_step_dist[env_ids], self.min_box_step_dist, self.max_box_step_dist
        )
        self.robot_from_box_dist[env_ids] = torch.clamp(
            self.robot_from_box_dist[env_ids], self.init_robot_box_dist, self.max_robot_box_dist
        )

    def box_from_step_dist_curriculum(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], randomize: bool = False):
        if self.box_from_step_dist is None:
            env.box_from_step_dist = torch.ones(env.num_envs, device=env.device) * self.min_box_step_dist

        self._update(env, env_ids)
        if randomize:
            env.box_from_step_dist[env_ids] = self.min_box_step_dist + torch.rand(len(env_ids), device=env.device) * (
                self.box_from_step_dist[env_ids] - self.min_box_step_dist
            )
        else:
            env.box_from_step_dist = self.box_from_step_dist

        return self.box_from_step_dist.mean().item()

    def robot_from_box_dist_curriculum(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], randomize: bool = False):
        if self.robot_from_box_dist is None:
            env.robot_from_box_dist = torch.ones(env.num_envs, device=env.device) * self.init_robot_box_dist

        self._update(env, env_ids)
        # we need to call update in each method, since the class instance is not shared between the methods

        if randomize:
            env.robot_from_box_dist[env_ids] = self.init_robot_box_dist + torch.rand(
                len(env_ids), device=env.device
            ) * (self.robot_from_box_dist[env_ids] - self.init_robot_box_dist)
        else:
            env.robot_from_box_dist = self.robot_from_box_dist

        return self.robot_from_box_dist.mean().item()
