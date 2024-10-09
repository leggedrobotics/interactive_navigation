# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from . import actions


@configclass
class MyJointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@configclass
class ArticulatedJumpActionCfg(ActionTermCfg):
    """Configuration for the jump action term.

    See :class:`ArticulatedJumpAction` for more details.
    """

    class_type: type[ActionTerm] = actions.ArticulatedJumpAction

    jump_cooldown_secs: float = 1.0
    """Cooldown time in seconds between jumps. Defaults to 1.0."""


@configclass
class ArticulatedWrench2DActionCfg(ActionTermCfg):
    """Configuration Wrench2D action term.

    See :class:`ArticulatedWrench2DAction` for more details.
    """

    max_velocity: float = 5.0

    max_rotvel: float = 2.0

    class_type: type[ActionTerm] = actions.ArticulatedWrench2DAction
