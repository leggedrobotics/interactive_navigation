from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .actions import SimpleAction


@configclass
class SimpleActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SimpleAction

    min_distance: float = 2.0
    """Minimum distance to the object to grab it."""

    fov_deg: float = 135.0
    """Horizontal field of view in degrees."""

    max_force: float = 2.0
    """Maximum force to apply to the robot."""

    max_toque: float = 1.0
    """Maximum torque to apply to the robot."""

    use_teleop: bool = True
    """Whether to use teleop interface for controlling the robot."""
