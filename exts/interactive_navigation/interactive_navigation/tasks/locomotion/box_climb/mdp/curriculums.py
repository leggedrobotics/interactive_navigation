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
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
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
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


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

        if "log" in env.extras:
            env.extras["log"]["Curriculum/successes"] = self.successes.mean().item()  # a bit hacky, but it works
            # env.extras["log"].update({"Curriculum/successes": self.successes.mean().item()})

        return torch.mean(terrain.terrain_levels.float()).item()
