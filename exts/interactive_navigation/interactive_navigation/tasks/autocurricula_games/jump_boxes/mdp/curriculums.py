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
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers import CurriculumTermCfg


import interactive_navigation.tasks.autocurricula_games.jump_boxes.mdp.actions as nav_actions

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class terrain_levels(ManagerTermBase):

    def __init__(
        self,
        cfg: CurriculumTermCfg,
        env: ManagerBasedRLEnv,
        num_successes: int = 5,
        num_failures: int = 5,
        goal_termination_name: str = "goal_reached",
        random_move_prob: float = 0.2,
    ):
        """Move up or down the terrain based on the number of successes and failures."""
        super().__init__(cfg, env)
        self.num_successes = num_successes
        self.num_failures = num_failures
        self.successes: torch.Tensor = None
        self.goal_termination_name = goal_termination_name
        self.random_move_prob = random_move_prob

    def __call__(
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

        if "log" in env.extras:
            env.extras["log"]["Curriculum/successes"] = self.successes.mean().item()  # a bit hacky, but it works
            # env.extras["log"].update({"Curriculum/successes": self.successes.mean().item()})

        return torch.mean(terrain.terrain_levels.float()).item()


def robot_speed_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_term_name: str,
    num_steps: int,
    start_multiplier: float = 2,
) -> float:
    """Curriculum to make the robot move slower. Slower is harder since the robot cannot jump very far.
    Only compatible with the ArticulatedWrench2DAction."""
    scaling_factor = start_multiplier - (start_multiplier - 1) * min(env.common_step_counter / num_steps, 1)

    action_term: nav_actions.ArticulatedWrench2DAction = env.action_manager._terms[action_term_name]  # type: ignore
    action_term.action_scale = scaling_factor
    action_term.max_lin_vel = action_term.cfg.max_velocity * scaling_factor
    action_term.max_vel_sideways = action_term.cfg.max_vel_sideways * scaling_factor
    action_term.max_rot_vel = action_term.cfg.max_rotvel * scaling_factor

    return scaling_factor


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


class entity_entity_dist_curriculum(ManagerTermBase):
    # TODO implement the curriculum for the distance between the robot and the box
    # increase distance on termination, when the reward new height is reached

    def __init__(
        self,
        cfg: CurriculumTermCfg,
        env: ManagerBasedRLEnv,
        start_dist: float = 2.0,
        max_dist: float = 4.0,
        dist_increment: float = 0.1,
        goal_termination_name: str = "goal_reached",
        move_down_factor: float = 2.0,
    ):
        super().__init__(cfg, env)

        self.goal_termination_name = goal_termination_name
        self.start_dist = start_dist
        self.max_dist = max_dist
        self.dist_increment = dist_increment
        self.move_down_factor = move_down_factor
        # buffers
        self.dist: torch.Tensor = None

    def _update(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]):

        if self.dist is None:
            self.dist = torch.ones(env.num_envs, device=env.device) * self.start_dist

        terminated_at_goal = env.termination_manager._term_dones[self.goal_termination_name]
        terminated = env.termination_manager.dones
        terminated_not_at_goal = terminated & ~terminated_at_goal
        if terminated_at_goal.any():
            # increase distance if goal was reached
            goal_reached_ids = env_ids[terminated_at_goal[env_ids]]
            self.dist[goal_reached_ids] += self.dist_increment
        if terminated_not_at_goal.any():
            # decrease distance if goal was not reached
            goal_not_reached_ids = env_ids[terminated_not_at_goal[env_ids]]
            self.dist[goal_not_reached_ids] -= self.dist_increment * self.move_down_factor

        # clamp the values
        self.dist[env_ids] = torch.clamp(self.dist[env_ids], self.start_dist, self.max_dist)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]):
        if self.dist is None:
            env.dist = torch.ones(env.num_envs, device=env.device) * self.start_dist
        self._update(env, env_ids)
        env.dist = self.dist
        return self.dist.mean().item()
