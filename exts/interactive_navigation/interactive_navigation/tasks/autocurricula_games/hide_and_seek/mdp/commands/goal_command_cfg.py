from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .goal_command import GoalCommand


@configclass
class GoalCommandCfg(CommandTermCfg):
    class_type = GoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    goal_radius_range: tuple[float, float] = (0.0, 0.0)
    """The goal will be sampled from a distance of the given range from the terrain origin."""

    only_heading: bool = False
    """If true, the goal is only the heading direction."""

    only_position: bool = False
    """If true, the goal is only the position."""

    show_line_to_goal: bool = True
    """If true, a line is drawn from the robot to the goal if debug vis is true."""

    show_goal_heading: bool = True
    """If true, an arrow is drawn to show the goal heading if debug vis is true."""

    env_frame: bool = False
    """If true, the gaol is returned in the environment frame, otherwise in the robot frame."""

    def __post_init__(self):
        self.goal_radius_range = tuple(self.goal_radius_range)
        super().__post_init__()
        self.show_goal_heading = self.show_goal_heading and not self.only_position
        self.show_line_to_goal = self.show_line_to_goal and not self.only_heading
