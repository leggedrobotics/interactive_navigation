from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, ARROW_X_MARKER_CFG

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

        self.low_level_policies = [self.climbing_policy, self.climbing_policy]

        self.num_skills = len(self.low_level_policies)

        # calculate decimation
        self.low_level_policy_decimation = int(1 / (cfg.locomotion_policy_freq * env.physics_dt))

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = 3 + 1  # [px,py, heading, omega] + selection: [action_1, action_2]

        # set up buffers
        self._init_buffers()

        # for policies trained with Isaac Gym, reorder the joint based on a provided list of joint names
        self.joint_mapping_gym_to_sim = env.scene["robot"].find_joints(
            env.scene["robot"].joint_names, self.cfg.reorder_joint_list, preserve_order=True
        )[0]
        # self.joint_mapping_sim_to_gym = [
        #     env.scene.articulations["robot"].joint_names.index(joint) for joint in self.cfg.reorder_joint_list
        # ]

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
        return self._raw_command_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """This returns the command for the low-level policies, predicted by the high-level policy."""
        return self._processed_command_actions

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
        self._raw_command_actions.copy_(action_command)
        # Create a mask for the skills
        for skill_id in range(self.num_skills):
            self._skill_mask[:, skill_id] = action_selection == skill_id

        # transform the actions

        if self.use_teleop:
            delta_pose, gripper_command = self.teleop_interface.advance()
            action_command[:, 0] = delta_pose[0]
            action_command[:, 1] = delta_pose[1]
            action_command[:, 2] = delta_pose[2]
            self._skill_mask[:, 0] = not gripper_command
            self._skill_mask[:, 1] = gripper_command

        squashed_actions = torch.tanh(action_command)

        position_2d = squashed_actions[:, :2] * 1.5  # position scaling TODO: make this a parameter
        angle = squashed_actions[:, 2] * torch.pi / 4  # angle scaling TODO: make this a parameter
        heading_sin_cos = torch.cat((torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)), dim=1)
        time_left = torch.ones_like(angle).unsqueeze(1) * 0.2  # time scaling TODO: make this a parameter or adaptive

        # ## debugging
        # position_2d = torch.zeros_like(position_2d)
        # position_2d[:, 0] = 4.0
        # time_left = self._env.episode_length_buf / self._env.max_episode_length
        # time_left = time_left.unsqueeze(1)
        # ## end debugging

        processed_actions = torch.cat((position_2d, heading_sin_cos, time_left), dim=1)

        # Apply the affine transformations
        self._processed_command_actions.copy_(processed_actions)

        # # For some skills (climbing), the command is a position and not a velocity
        # # To account for this, we multiply the velocity command by N time steps (carrot on a stick)
        # self._processed_command_actions[self._skill_mask[:, 1]] *= self.low_level_policy_decimation

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.low_level_policy_decimation == 0:
            # update low-level action at 50Hz
            self._counter = 0

            # Get low level actions from low level policy
            print(f"skill {self._skill_mask[0].nonzero().item()}")

            for skill_id in range(self.num_skills):
                self.low_level_actions[self._skill_mask[:, skill_id]] = self.low_level_policies[skill_id](
                    self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)[
                        self._skill_mask[:, skill_id]  # type: ignore
                    ]
                )
            self._prev_low_level_actions.copy_(self._low_level_actions.clone())
            # self._low_level_actions.copy_(
            #     self.locomotion_policy(
            #         self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
            #     )
            # )

            if self.cfg.reorder_joint_list:
                self._low_level_actions = self._low_level_actions[:, self.joint_mapping_gym_to_sim]

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
        self._raw_command_actions = torch.zeros(self.num_envs, self._action_dim - 1, device=self.device)
        self._processed_command_actions = torch.zeros((self.num_envs, 5), device=self.device)
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
            if not hasattr(self, "pos_cmd_visualizer"):
                marker_cfg = ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["blue_arrow"].scale = (0.1, 0.1, 1.0)
                marker_cfg.markers["red_arrow"].scale = (0.1, 0.1, 1.0)
                marker_cfg.prim_path = "/Visuals/Command/action_force"

                self.pos_cmd_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "head_cmd_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/action_force"

                self.head_cmd_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.pos_cmd_visualizer.set_visibility(True)
            self.head_cmd_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- goal pose

        # - get robot pos
        robot_pos = self._env.scene[self.robot_name].data.root_pos_w.clone()
        robot_quat = math_utils.yaw_quat(self._env.scene[self.robot_name].data.root_quat_w)
        # increase z to visualize the force
        viz_pos_offset = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        # - get command (velocity)
        command = self.processed_actions
        pos_command = command[:, :2]
        heading_command = command[:, 2:4]
        pos_command_3d = torch.cat([pos_command, torch.zeros(self.num_envs, device=self.device).unsqueeze(1)], dim=1)
        head_command_3d = torch.cat(
            [
                heading_command[:, 1:2],
                heading_command[:, 0:1],
                torch.zeros(self.num_envs, device=self.device).unsqueeze(1),
            ],
            dim=1,
        )
        # omega_z_command = command[:, 2] # TODO, visualize omega_z
        pos_command_w = math_utils.quat_apply(robot_quat, pos_command_3d)
        head_command_w = math_utils.quat_apply(robot_quat, head_command_3d)

        yaw_angle = torch.atan2(pos_command_w[:, 1], pos_command_w[:, 0])
        yaw_angle_heading_w = torch.atan2(head_command_w[:, 1], head_command_w[:, 0])

        pos_dir_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_angle), torch.zeros_like(yaw_angle), yaw_angle
        )
        head_dir_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_angle_heading_w), torch.zeros_like(yaw_angle_heading_w), yaw_angle_heading_w
        )

        scales = torch.linalg.norm(pos_command, dim=1, keepdim=True)
        default_scale = torch.ones_like(scales) * 2
        scales_3d = torch.cat([scales, default_scale, default_scale], dim=1)
        scales_3d[scales_3d == 0] = 0.1
        const_scales = torch.cat([default_scale * 4, default_scale, default_scale], dim=1)

        self.pos_cmd_visualizer.visualize(
            translations=robot_pos + viz_pos_offset * 0.4,
            orientations=pos_dir_quat,
            scales=scales_3d,
            marker_indices=torch.argmax(self._skill_mask.int(), dim=1),
        )
        self.head_cmd_visualizer.visualize(
            translations=robot_pos + viz_pos_offset * 0.2, orientations=head_dir_quat, scales=const_scales
        )
