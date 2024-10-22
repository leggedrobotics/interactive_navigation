# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.commands import GoalCommand

from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.utils import (
    get_robot_pos,
    get_robot_quat,
    get_robot_lin_vel_w,
    get_robot_rot_vel_w,
)


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def too_far_away(
    env: ManagerBasedRLEnv,
    max_dist: float,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot is too far away from the goal."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    goal_cmd_geneator: GoalCommand = env.command_manager._terms[command_name]

    robot_pos = get_robot_pos(robot)
    goal_pos = goal_cmd_geneator.goal_pos_w

    diff = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
    return diff > max_dist
