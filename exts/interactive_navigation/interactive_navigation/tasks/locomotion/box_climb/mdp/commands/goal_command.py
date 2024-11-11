from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING
from omni.isaac.lab.markers.config import (
    FRAME_MARKER_CFG,
    CUBOID_MARKER_CFG,
    CYLINDER_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)
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

        # - positions
        # goal position in world frame
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)  # to be sampled
        # goal position in robot frame
        self.goal_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        # sample potential goals

        # - heading
        # goal heading in world frame. This is THE goal heading angle in the world frame
        self.goal_heading = torch.zeros((self.num_envs,), device=self.device)
        self.heading_error_angle = torch.zeros((self.num_envs,), device=self.device)
        # heading of the robot (cos(yaw), sin(yaw)).
        self.heading_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.show_line_to_goal = cfg.show_line_to_goal

    def __str__(self) -> str:
        return f"Height Goal Command for {self.cfg.asset_name}"

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The current goal state g(s_t).
        Future samples of this are used to train the actor and critic.
        This is also used for observations."""
        # position (3d) and heading (2d)

        dist = torch.linalg.vector_norm(self.goal_pos_b, dim=1).unsqueeze(1) + 1e-6
        rescaled_goal_pos_b = self.goal_pos_b / dist * torch.log(dist + 1)

        if self.cfg.heading:
            return torch.cat([rescaled_goal_pos_b, self.heading_error], dim=1)
        return rescaled_goal_pos_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics for the goal command."""
        # self.metrics["goal_height_w"] = self.goal_pos_w[:, 2].clone()
        self.metrics["goal_distance"] = torch.linalg.vector_norm(self.goal_pos_b, dim=1)
        if self.cfg.heading:
            self.metrics["goal_height_b"] = self.goal_pos_b[:, 2].clone()

    def _resample_command(self, env_ids: Sequence[int]):
        """The goal is to reach the max height of the current terrain."""
        # - goal position
        self.goal_pos_w[env_ids] = self.env.scene.terrain.env_origins[env_ids]
        # increase z position by the robot height
        self.goal_pos_w[env_ids, 2] += self.robot_height

        # - goal heading
        if self.cfg.heading:
            self.goal_heading[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi

    def _update_command(self):
        """The goal pose has to be updated to be in the robots frame"""

        # get the current pos of the robot
        robot_pos_w = self.robot.data.root_pos_w.clone()
        robot_quat = self.robot.data.root_quat_w.clone()
        self.goal_pos_b, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_quat, self.goal_pos_w)

        # get heading
        if self.cfg.heading:
            yaw_angle = math_utils.euler_xyz_from_quat(robot_quat)[2]
            self.heading_error_angle = math_utils.wrap_to_pi(self.goal_heading - yaw_angle)
            self.heading_error = torch.stack(
                [torch.cos(self.heading_error_angle), torch.sin(self.heading_error_angle)], dim=1
            )

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
                marker_cfg.markers["cylinder"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "goal_heading_visualizer") and self.cfg.heading:
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_heading"
                marker_cfg.markers["arrow"].scale = ((0.4, 0.4, 1.0),)

                # marker_cfg.markers["arrow"].height = 0.5
                # marker_cfg.markers["arrow"].radius = 0.05
                # marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                self.goal_heading_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
            if self.cfg.show_line_to_goal:
                self.line_to_goal_visualiser.set_visibility(True)
            if self.cfg.show_goal_heading and self.cfg.heading:
                self.goal_heading_visualizer.set_visibility(True)
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
        line_goal_pos = self.goal_pos_w.clone()
        line_goal_pos[:, 2] += 0.5  # for better visaibility
        difference = line_goal_pos - self.robot.data.root_pos_w.clone()
        translations = self.robot.data.root_pos_w.clone()
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
        axis = torch.linalg.cross(difference, x_vec.expand_as(difference))
        quat = math_utils.quat_from_angle_axis(angle, axis)
        # apply transforms
        if self.cfg.show_line_to_goal:
            self.line_to_goal_visualiser.visualize(translations=translations, scales=scales, orientations=quat)

        # update the goal heading marker
        if self.cfg.heading and self.cfg.show_goal_heading:
            goal_quat = math_utils.quat_from_angle_axis(
                self.goal_heading, torch.tensor([0.0, 0.0, 1.0]).to(self.device)
            )
            self.goal_heading_visualizer.visualize(translations=line_goal_pos, orientations=goal_quat)
