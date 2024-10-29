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


class TerrainCurriculum:

    def __init__(
        self,
        num_successes: int = 5,
        num_failures: int = 5,
        goal_termination_name: str = "goal_reached",
        random_move_prob: float = 0.2,
    ):
        """Move up or down the terrain based on the number of successes and failures."""
        self.num_successes = num_successes
        self.num_failures = num_failures
        self.successes: torch.Tensor = None
        self.goal_termination_name = goal_termination_name
        self.random_move_prob = random_move_prob

    def terrain_levels(
        self, env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> float:
        """Curriculum based on goal reached"""
        if self.successes is None:
            self.successes = torch.zeros(env.num_envs, device=env.device)

        terminated_at_goal = env.termination_manager._term_dones[self.goal_termination_name]
        terminated = env.termination_manager.dones
        terminated_not_at_goal = terminated & ~terminated_at_goal

        self.successes[env_ids] += terminated_at_goal[env_ids].float()
        self.successes[env_ids] -= terminated_not_at_goal[env_ids].float()

        move_up = self.successes >= self.num_successes
        move_down = self.successes <= -self.num_failures
        # reset the successes and failures to zero
        self.successes[move_up | move_down] = 0

        # randomly move up or down
        random_move = torch.rand_like(move_up.float()) < self.random_move_prob
        random_move_up = torch.rand_like(move_up.float()) < 0.5
        move_up |= random_move & random_move_up
        move_down |= random_move & ~random_move_up

        # update terrain levels
        terrain: TerrainImporter = env.scene.terrain
        terrain.update_env_origins(env_ids, move_up[env_ids], move_down[env_ids])
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
