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
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.utils import (
    get_robot_pos,
    get_robot_quat,
    get_robot_lin_vel_w,
    get_robot_rot_vel_w,
)

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
        self.goal_pos_env = torch.zeros((self.num_envs, 3), device=self.device)  # in env/terrain frame
        self.robot_pos_env = torch.zeros((self.num_envs, 3), device=self.device)  # in env/terrain frame
        # robot pos in robot frame is not necessary, would be identical to 0

        # - heading
        # goal heading in world frame. This is THE goal heading angle in the world frame
        self.goal_heading = torch.zeros((self.num_envs,), device=self.device)
        ideal_heading_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.goal_heading_error = torch.cat([torch.cos(ideal_heading_error), torch.sin(ideal_heading_error)], dim=1)
        # heading of the robot (cos(yaw), sin(yaw)).
        self.heading_error = torch.zeros((self.num_envs, 2), device=self.device)

        # - attitude
        self.attitude = torch.zeros((self.num_envs,), device=self.device)

        assert not (
            self.cfg.only_heading and self.cfg.only_position
        ), "Cannot have both only_heading and only_position set to True"

    def __str__(self) -> str:
        return f"Height Goal Command for {self.cfg.asset_name}"

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """We abuse the command to return the goal position and heading error. We use the
        goal property to condition the policy
        The current goal state g(s_t).
        Future samples of this are used to train the actor and critic.
        This is also used for observations."""
        # Current position (3d) and heading (2d)

        if self.cfg.only_heading:
            return self.heading_error
        elif self.cfg.only_position:
            if self.cfg.env_frame:
                # current robot pos in env frame
                current_goal = self.robot_pos_env[..., :2]
            else:
                current_goal = self.goal_pos_b[..., :2]
        else:
            current_goal = torch.cat([self.goal_pos_b[..., :2], self.heading_error], dim=1)

        if self.cfg.attitude:
            current_goal = torch.cat([current_goal, self.attitude.unsqueeze(1)], dim=1)
        return current_goal

    @property
    def goal(self) -> torch.Tensor:
        """The desired goal state g(s*).
        This is used to condition the policy during rollouts.
        This value is NOT being recorded to the buffer."""
        # this is zero if in robot frame because the desired goal pos in the robot frame is always 0 = at the goal
        # the goal heading error is 0 --> cos sin of 0 = 1, 0

        if self.cfg.only_heading:
            return self.goal_heading_error
        elif self.cfg.only_position:
            if self.cfg.env_frame:
                des_goal = self.goal_pos_env[..., :2]
            else:
                des_goal = torch.zeros_like(self.goal_pos_b[..., :2])

        elif self.cfg.env_frame:
            des_goal = torch.cat([self.goal_pos_env[..., :2], self.goal_heading_error], dim=1)
        else:
            des_goal = torch.cat([torch.zeros_like(self.goal_pos_b[..., :2]), self.goal_heading_error], dim=1)

        if self.cfg.attitude:
            des_goal = torch.cat([des_goal, torch.zeros_like(self.attitude).unsqueeze(1)], dim=1)

        return des_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics for the goal command."""
        if not self.cfg.only_heading:
            self.metrics["dist_to_goal"] = torch.linalg.norm(self.goal_pos_b[:, :2], dim=1)
        if not self.cfg.only_position:
            self.metrics["heading_error"] = torch.acos(self.heading_error[:, 0])
        if self.cfg.attitude:
            self.metrics["attitude"] = self.attitude

    def _resample_command(self, env_ids: Sequence[int]):
        """The goal is to reach the max height of the current terrain."""
        # - goal position
        # get potential goals
        terrain = self.env.scene.terrain

        # goal positions
        random_distance = (
            torch.rand(len(env_ids), device=self.device)
            * (self.cfg.goal_radius_range[1] - self.cfg.goal_radius_range[0])
            + self.cfg.goal_radius_range[0]
        )
        random_angle = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        random_offset = torch.stack(
            [torch.cos(random_angle), torch.sin(random_angle)], dim=1
        ) * random_distance.unsqueeze(1)
        self.goal_pos_env[env_ids, :2] = random_offset
        self.goal_pos_w[env_ids] = terrain.env_origins[env_ids]
        self.goal_pos_w[env_ids, :2] += random_offset

        # - goal heading
        self.goal_heading[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi

    def _update_command(self):
        """The goal pose has to be updated to be in the robots frame"""

        # get the current pos of the robot
        robot_pos_w = get_robot_pos(self.robot)
        robot_pos_w[:, 2] -= self.robot_height
        robot_quat = math_utils.yaw_quat(get_robot_quat(self.robot))

        self.goal_pos_b, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_quat, self.goal_pos_w)

        if self.cfg.env_frame:
            terrain_origins = self.env.scene.terrain.env_origins
            self.robot_pos_env = robot_pos_w - terrain_origins

        # get heading
        yaw_angle = math_utils.euler_xyz_from_quat(robot_quat)[2]
        heading_error = self.goal_heading - yaw_angle
        self.heading_error = torch.stack([torch.cos(heading_error), torch.sin(heading_error)], dim=1)

        # get attitude
        if self.cfg.attitude:
            self.attitude = torch.acos(-torch.clamp(self.robot.data.projected_gravity_b[:, 2], -1, 1))

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
            if not hasattr(self, "goal_heading_visualizer"):
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
            else:
                self.line_to_goal_visualiser.set_visibility(False)
            if self.cfg.show_goal_heading:
                self.goal_heading_visualizer.set_visibility(True)
            else:
                self.goal_heading_visualizer.set_visibility(False)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if hasattr(self, "line_to_goal_visualiser"):
                self.line_to_goal_visualiser.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback function for the debug visualization."""

        # update the line marker
        # calculate the difference vector between the robot root position and the goal position
        line_goal_pos = self.goal_pos_w.clone()
        line_goal_pos[:, 2] += 0.5  # for better visaibility
        difference = line_goal_pos - get_robot_pos(self.robot)
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
        axis = torch.linalg.cross(difference, x_vec.expand_as(difference))
        quat = math_utils.quat_from_angle_axis(angle, axis)
        # apply transforms
        self.line_to_goal_visualiser.visualize(translations=translations, scales=scales, orientations=quat)

        # update the goal heading marker
        line_goal_pos[:, 2] += 1.0
        goal_quat = math_utils.quat_from_angle_axis(self.goal_heading, torch.tensor([0.0, 0.0, 1.0]).to(self.device))
        self.goal_heading_visualizer.visualize(translations=line_goal_pos, orientations=goal_quat)

        # update the box marker
        self.box_goal_visualizer.visualize(self.goal_pos_w)

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
