from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from .interactive_navigation_action_cfg import InteractiveNavigationActionCfg


class InteractiveNavigationAction(ActionTerm):

    cfg: InteractiveNavigationActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: InteractiveNavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_name = "robot"

        # -- load low level policies
        if not check_file_path(cfg.locomotion_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.locomotion_policy_file}' does not exist.")
        # load policies
        file_bytes = read_file(self.cfg.locomotion_policy_file)
        self.locomotion_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.locomotion_policy = torch.jit.freeze(self.locomotion_policy.eval())

        # calculate decimation
        self.locomotion_policy_decimation = int(1 / (cfg.locomotion_policy_freq * env.physics_dt))

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = 3  # [vx, vy, omega] # TODO: policy selection with categorical action space

        # set up buffers
        self._init_buffers()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """This returns the command for the low-level policies, predicted by the high-level policy."""
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    @property
    def prev_low_level_actions(self) -> torch.Tensor:
        return self._prev_low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process high-level actions. This function is called with a frequency of 10Hz.

        Args:
            actions (torch.Tensor): The high level action command to condition the low-level policies.
        """

        # Depending on the action distribution, the actions need to be processed differently

        # Store the raw low-level navigation actions
        self._raw_navigation_velocity_actions.copy_(actions)

        squashed_actions = torch.tanh(actions)

        # Apply the affine transformations
        self._processed_navigation_velocity_actions.copy_(squashed_actions * self._scale + self._offset)

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.locomotion_policy_decimation == 0:
            # update low-level action at 50Hz
            self._counter = 0
            self._prev_low_level_actions.copy_(self._low_level_actions.clone())
            # Get low level actions from low level policy
            self._low_level_actions.copy_(
                self.locomotion_policy(
                    self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
                )
            )
            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._prev_low_level_actions = torch.zeros_like(self._low_level_actions)
        self._counter = 0
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "force_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/action_force"

                self.force_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.force_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- goal pose

        # - get robot pos
        robot_pos = self._env.scene[self.robot_name].data.root_pos_w.clone()
        robot_quat = math_utils.yaw_quat(self._env.scene[self.robot_name].data.root_quat_w)
        # increase z to visualize the force
        robot_pos[:, 2] += 1.0

        # - get command (velocity)
        command = self.processed_actions
        velocity_command = command[:, :2]
        velocity_command_3d = torch.cat(
            [velocity_command, torch.zeros(self.num_envs, device=self.device).unsqueeze(1)], dim=1
        )
        # omega_z_command = command[:, 2] # TODO, visualize omega_z
        velocity_command_w = math_utils.quat_apply(robot_quat, velocity_command_3d)

        yaw_angle = torch.atan2(velocity_command_w[:, 1], velocity_command_w[:, 0])

        vel_dir_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_angle), torch.zeros_like(yaw_angle), yaw_angle
        )

        scales = torch.linalg.norm(velocity_command, dim=1, keepdim=True) * 12
        default_scale = torch.ones_like(scales) * 4
        scales_3d = torch.cat([scales * 2, default_scale, default_scale], dim=1)

        self.force_visualizer.visualize(translations=robot_pos, orientations=vel_dir_quat, scales=scales_3d)
