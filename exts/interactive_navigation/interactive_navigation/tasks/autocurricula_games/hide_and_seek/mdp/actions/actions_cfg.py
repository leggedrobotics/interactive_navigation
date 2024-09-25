from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .actions import SimpleAction


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

    use_teleop: bool = True
    """Whether to use teleop interface for controlling the robot."""
