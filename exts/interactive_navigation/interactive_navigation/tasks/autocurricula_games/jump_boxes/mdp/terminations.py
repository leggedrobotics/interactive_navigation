from __future__ import annotations

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import TerminationTermCfg

from interactive_navigation.tasks.autocurricula_games.jump_boxes.mdp.commands import GoalCommand


def goal_reached(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.
    Args:
        env: The learning environment.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    distance_goal = torch.linalg.vector_norm(goal_cmd_geneator.goal_pos_b, dim=1)
    # Check conditions
    at_goal = (distance_goal < distance_threshold) & (torch.abs(goal_cmd_geneator.goal_pos_b[:, 2]) < 0.1)
    return at_goal


def too_far_from_goal(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 50.0,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the robot is too far from the goal.
    Args:
        env: The learning environment.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the robot is too far from the goal.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    distance_goal = torch.linalg.vector_norm(goal_cmd_geneator.goal_pos_b, dim=1)
    # Check conditions
    too_far = distance_goal > distance_threshold
    return too_far


class time_out_perturbed(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.perturbation_per_env = None

    def __call__(self, env: ManagerBasedRLEnv, perturbation: float, only_longer: bool = True) -> torch.Tensor:
        """Terminate the episode when the episode length exceeds the maximum episode length, with a perturbation
        to prevent all episodes from terminating simultaneously.
        Args:
            env: The learning environment.
            perturbation: The perturbation to the maximum episode length.
            only_longer: If True, the perturbation is always positive, i.e., episodes are longer then the maximum episode
        """

        if self.perturbation_per_env is None:
            self.perturbation_per_env = torch.rand(env.num_envs).to(env.device) * perturbation / env.step_dt
        if env.termination_manager.dones.any():
            # update the perturbation for the terminated environments
            self.perturbation_per_env[env.termination_manager.dones] = (
                torch.rand(env.termination_manager.dones.sum()).to(env.device) * perturbation / env.step_dt
            )

        if not only_longer:
            self.perturbation_per_env = self.perturbation_per_env * 2 - perturbation / env.step_dt
        return env.episode_length_buf >= env.max_episode_length + self.perturbation_per_env
