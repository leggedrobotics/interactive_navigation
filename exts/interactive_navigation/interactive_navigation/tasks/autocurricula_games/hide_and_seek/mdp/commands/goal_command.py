from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG, CYLINDER_MARKER_CFG
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.utils.warp import raycast_mesh
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils import math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from .goal_command_cfg import GoalCommandCfg


class GoalCommand(CommandTerm):
    """Command for the goal height of the robot.
    The goal is to reach the top of the terrain.
    It is implemented as the difference between the current height of the robot and the max height of the terrain.

    The desired goal thus is always 0.

    """

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)
        self.env = env
        self.cfg = cfg
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.robot_height = 0.3  # half height of the robot

        # goal height w is constant, ie always on the top of the terrain
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # goal height b is the current height of the robot
        self.goal_pos_b = torch.zeros((self.num_envs, 3), device=self.device)

        self.show_line_to_goal = True

        # sample potential goals
        self.highest_positions = self._resample_highest_pos()

    def __str__(self) -> str:
        return f"Height Goal Command for {self.cfg.asset_name}"

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The current goal state g(s_t)."""
        return self.goal_pos_b

    @property
    def goal(self) -> torch.Tensor:
        """The desired goal state g(s*)."""
        return self.goal_pos_w  # always 0

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics for the goal command."""
        self.metrics["goal_height_w"] = self.goal_pos_w[:, 2]
        self.metrics["goal_height_b"] = self.goal_pos_b[:, 2]

    def _resample_command(self, env_ids: Sequence[int]):
        """The goal is to reach the max height of the current terrain."""
        # get potential goals
        terrain = self.env.scene.terrain
        potential_goals = self.highest_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]

        # select random goal:
        random_indices = torch.randint(0, potential_goals.shape[1], (len(env_ids),))
        new_goals = potential_goals[torch.arange(len(env_ids)), random_indices]
        self.goal_pos_w[env_ids] = new_goals

    def _update_command(self):
        """The goal height has to be updated to be in the robots frame.
        Its simply the difference between the goal height and the current height of the robot."""

        # get the current height of the robot
        robot_heights_w = get_robot_pos(self.robot) - self.robot_height
        self.goal_pos_b = robot_heights_w - self.goal_pos_w

    """
    Debug Visualizations
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "line_to_goal_visualiser"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/line_to_goal"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.05
                self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
            if self.cfg.show_line_to_goal:
                self.line_to_goal_visualiser.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if hasattr(self, "line_to_goal_visualiser"):
                self.line_to_goal_visualiser.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback function for the debug visualization."""
        # update the box marker
        self.box_goal_visualizer.visualize(self.goal_pos_w)

        # update the line marker
        # calculate the difference vector between the robot root position and the goal position
        difference = self.goal_pos_w - get_robot_pos(self.robot)
        translations = get_robot_pos(self.robot).clone()
        # calculate the scale of the arrow (Mx3)
        scales = torch.norm(difference, dim=1)
        # translate half of the length along difference axis
        translations += difference / 2
        # scale along x axis
        scales = torch.vstack([scales, torch.ones_like(scales), torch.ones_like(scales)]).T
        # convert the difference vector to a quaternion
        difference = torch.nn.functional.normalize(difference, dim=1)
        x_vec = torch.tensor([1, 0, 0]).float().to(self.device)
        angle = -torch.acos(difference @ x_vec)
        axis = torch.cross(difference, x_vec.expand_as(difference))
        quat = math_utils.quat_from_angle_axis(angle, axis)
        # apply transforms
        self.line_to_goal_visualiser.visualize(translations=translations, scales=scales, orientations=quat)

    ##
    #  Utility functions
    ##

    def _resample_highest_pos(self):
        """This function samples the highest points on the terrain for each terrain in the environment.
        It uses raycasting to find the highest points on the terrain."""

        N_points_per_terrain = 25

        # get terrain
        terrain = self.env.scene.terrain
        wp_terrain_mesh = terrain.warp_meshes["terrain"]
        # env_origin = terrain.env_origins[env_ids]
        rows, cols, _ = terrain.terrain_origins.shape

        highest_positions = torch.zeros((rows, cols, N_points_per_terrain, 3), device=self.device)

        # create rays:
        pattern = patterns.GridPatternCfg(
            size=terrain.cfg.terrain_generator.size,
            resolution=0.25,
        )
        ray_starts, ray_directions = pattern.func(pattern, self.device)
        ray_starts[..., 2] = 50  # start from a high point

        for row in range(rows):
            for col in range(cols):
                # raycast the terrain from above without the borders
                terrain_origin = terrain.terrain_origins[row, col]
                sub_ray_starts = ray_starts + terrain_origin
                ray_hits = raycast_mesh(sub_ray_starts.unsqueeze(0), ray_directions.unsqueeze(0), wp_terrain_mesh.id)[
                    0
                ].squeeze(0)
                # get the the highes point on the terrain
                non_border_ray_hits = ray_hits[ray_hits[..., 2] < terrain.cfg.terrain_generator.border_height - 0.1]
                top_hits = non_border_ray_hits[non_border_ray_hits[..., 2] > non_border_ray_hits[..., 2].max() - 0.1]

                # randomly select N_points_per_terrain of the top hits
                if top_hits.shape[0] > N_points_per_terrain:
                    indices = torch.randperm(top_hits.shape[0])[:25]
                    top_hits = top_hits[indices]
                else:
                    raise ValueError(
                        f"Not enough top hits found in the terrain for row {row} and col {col}. Found {top_hits.shape[0]} hits. Required {N_points_per_terrain}"
                    )
                highest_positions[row, col] = top_hits

        return highest_positions


##
#  Utility functions
##


def get_robot_pos(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the position of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_pos_w[:, -1, :]
    else:
        return robot.data.root_pos_w
