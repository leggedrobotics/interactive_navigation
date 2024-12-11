from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from .interactive_navigation_action import InteractiveNavigationAction


@configclass
class InteractiveNavigationActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = InteractiveNavigationAction
    """ Class of the action term."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    locomotion_policy_file: str = MISSING
    """Path to the low level locomotion policy file."""
    climbing_policy_file: str = MISSING
    """Path to the low level climbing policy file."""
    locomotion_policy_freq: float = MISSING
    """Frequency of the low level locomotion policy."""
    observation_group: str = MISSING
    """Observation group to use for the low level policy."""
    reorder_joint_list: list[str] = MISSING
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order."""
