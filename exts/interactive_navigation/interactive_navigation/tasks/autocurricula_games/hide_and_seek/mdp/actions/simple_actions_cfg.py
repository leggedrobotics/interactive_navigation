from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .simple_actions import SimpleAction, WrenchAction2D, JumpAction


@configclass
class SimpleActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SimpleAction

    min_distance: float = 1.5
    """Minimum distance to the object to grab it (center to center)."""

    fov_deg: float = 135.0
    """Horizontal field of view in degrees."""

    max_force: float = 5.0
    """Maximum force to apply to the robot."""

    max_lin_vel: float = 2.5
    """Maximum linear velocity of the robot in m/s."""

    max_toque: float = 5.0
    """Maximum torque to apply to the robot."""

    jump_cooldown_secs: float = 1.0
    """Cooldown time in seconds for the jump action."""


@configclass
class WrenchAction2DCfg(ActionTermCfg):
    class_type: type[ActionTerm] = WrenchAction2D

    max_force_forward: float = 5.0
    """Maximum force to apply to the robot."""

    max_foce_sideways: float = 3.5
    """Maximum force to apply to the robot."""

    max_torque: float = 5.0
    """Maximum torque to apply to the robot."""

    max_lin_vel: float = 2.5
    """Maximum linear velocity of the robot in m/s."""

    max_ang_vel: float = 3.14
    """Maximum angular velocity of the robot in rad/s."""


@configclass
class JumpActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = JumpAction

    jump_height: float = 1.0
    """Height of the jump in meters."""

    jump_cooldown_secs: float = 1.0
    """Cooldown time in seconds for the jump action."""
