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


TELEOP = True
if TELEOP:
    from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse, Se3Gamepad


class InteractiveNavigationAction(ActionTerm):

    cfg: InteractiveNavigationActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: InteractiveNavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_name = "robot"

        # -- load low level policies
        if not check_file_path(cfg.locomotion_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.locomotion_policy_file}' does not exist.")
        if not check_file_path(cfg.climbing_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.climbing_policy_file}' does not exist.")
        # load policies
        file_bytes = read_file(self.cfg.locomotion_policy_file)
        self.locomotion_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.locomotion_policy = torch.jit.freeze(self.locomotion_policy.eval())
        file_bytes = read_file(self.cfg.climbing_policy_file)
        self.climbing_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.climbing_policy = torch.jit.freeze(self.climbing_policy.eval())

        self.low_level_policies = [self.locomotion_policy, self.climbing_policy]

        self.num_skills = len(self.low_level_policies)

        # calculate decimation
        self.low_level_policy_decimation = int(1 / (cfg.locomotion_policy_freq * env.physics_dt))

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = 3 + 1  # [vx, vy, omega] + selection: [action_1, action_2]

        # set up buffers
        self._init_buffers()

        # teleop:
        self.use_teleop = TELEOP
        if self.use_teleop:
            self.teleop_interface = Se3Keyboard(pos_sensitivity=1, rot_sensitivity=1)
            self.teleop_interface.add_callback("L", env.reset)
            print(self.teleop_interface)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_velocity_command_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """This returns the command for the low-level policies, predicted by the high-level policy."""
        return self._processed_velocity_command_actions

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
        action_command = actions[:, :3]
        action_selection = actions[:, 3:].squeeze(1)

        # Store the raw low-level navigation actions
        self._raw_velocity_command_actions.copy_(action_command)
        # Create a mask for the skills
        for skill_id in range(self.num_skills):
            self._skill_mask[:, skill_id] = action_selection == skill_id

        squashed_actions = torch.tanh(action_command)
        # Apply the affine transformations
        self._processed_velocity_command_actions.copy_(squashed_actions * self._scale + self._offset)

        # For some skills (climbing), the command is a position and not a velocity
        # To account for this, we multiply the velocity command by N time steps (carrot on a stick)
        self._processed_velocity_command_actions[self._skill_mask[:, 1]] *= self.low_level_policy_decimation

        if self.use_teleop:
            delta_pose, gripper_command = self.teleop_interface.advance()
            action_command[:, 0] = delta_pose[0]
            action_command[:, 1] = delta_pose[1]
            action_command[:, 2] = delta_pose[2]
            self._processed_velocity_command_actions.copy_(action_command * self._scale)
            self._skill_mask[:, 0] = not gripper_command
            self._skill_mask[:, 1] = gripper_command

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.low_level_policy_decimation == 0:
            # update low-level action at 50Hz
            self._counter = 0
            self._prev_low_level_actions.copy_(self._low_level_actions.clone())
            # Get low level actions from low level policy

            for skill_id in range(self.num_skills):
                self.low_level_actions[self._skill_mask[:, skill_id]] = self.low_level_policies[skill_id](
                    self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)[
                        self._skill_mask[:, skill_id]  # type: ignore
                    ]
                )

            # self._low_level_actions.copy_(
            #     self.locomotion_policy(
            #         self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
            #     )
            # )

            # Process low level actions
            self.low_level_action_term.process_actions(
                self._low_level_actions
            )  # assuming all low level skills have the same action

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_velocity_command_actions = torch.zeros(self.num_envs, self._action_dim - 1, device=self.device)
        self._processed_velocity_command_actions = torch.zeros(
            (self.num_envs, self._action_dim - 1), device=self.device
        )
        self._skill_mask = torch.zeros(self.num_envs, self.num_skills, device=self.device).bool()
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
