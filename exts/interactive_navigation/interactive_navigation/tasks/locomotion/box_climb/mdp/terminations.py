from __future__ import annotations

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import TerminationTermCfg

from interactive_navigation.tasks.locomotion.box_climb.mdp.commands import GoalCommand


def goal_reached_pos_only(
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


def goal_reached(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.5,
    heading_threshold: float = 0.1,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the goal state is reached.
    Args:
        env: The learning environment.
        distance_threshold: The distance threshold to the goal.
        heading_threshold: The heading threshold to the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_generator: GoalCommand = env.command_manager._terms[goal_cmd_name]
    # check for termination
    dist = goal_cmd_generator.command[:, :3]
    heading = goal_cmd_generator.heading_error_angle

    goal_reached = torch.linalg.vector_norm(dist, dim=1) < distance_threshold
    heading_reached = torch.abs(heading) < heading_threshold

    at_goal = goal_reached & heading_reached

    return at_goal
