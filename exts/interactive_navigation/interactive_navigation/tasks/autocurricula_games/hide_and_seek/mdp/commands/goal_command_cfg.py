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
