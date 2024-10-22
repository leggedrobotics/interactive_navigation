from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .goal_command import GoalCommand


@configclass
class GoalCommandCfg(CommandTermCfg):
    class_type = GoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    show_line_to_goal: bool = True
    """If true, a line is drawn from the robot to the goal if debug vis is true."""

    show_goal_heading: bool = True
    """If true, an arrow is drawn to show the goal heading if debug vis is true."""

    randomize_goal: bool = False
    """If true, the goal is randomized at the start of each episode."""

    only_heading: bool = False
    """If true, the goal is only the heading direction."""
