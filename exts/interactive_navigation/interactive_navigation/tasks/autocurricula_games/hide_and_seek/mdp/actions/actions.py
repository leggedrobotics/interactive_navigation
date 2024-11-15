# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import math

TELEOP = False

if TELEOP:
    from omni.isaac.lab.devices import Se3Keyboard
import carb

import omni.isaac.lab.utils.string as string_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from . import actions_cfg


class MyJointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.MyJointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.MyJointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class ArticulatedJumpAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands.
    The jump actions simply actuates the z joint until the robot reaches a certain height."""

    cfg: actions_cfg.ArticulatedJumpActionCfg
    _env: ManagerBasedRLEnv
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.ArticulatedJumpActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.robot_name = "robot"
        self.env = env

        # jump config
        self.jump_cooldown_steps = int(math.ceil(cfg.jump_cooldown_secs / self.env.step_dt))
        jump_height = 0.85  # meters
        g = 9.81  # m/s^2
        self.jump_velocity = math.sqrt(2 * g * jump_height)

        self.jump_cooldown_buffer = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device)
        self.jumping_now = torch.zeros(self.env.num_envs, dtype=torch.bool).to(self.env.device)
        self.bottom_height = torch.zeros(self.env.num_envs).to(self.env.device)

        self.des_vel = torch.zeros_like(self._asset.data.body_lin_vel_w[:, -1, :])

        if TELEOP:
            self.teleop_interface = Se3Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
            self.teleop_interface.add_callback("L", env.reset)
            print(self.teleop_interface)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return torch.empty()

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    def process_actions(self, actions: torch.Tensor):
        """called each env step, i.e., once per {decimation} sim steps"""
        # set joint effort targets
        if TELEOP:
            delta_pose, gripper_command = self.teleop_interface.advance()
            want_to_jump = (torch.zeros_like(actions.squeeze()) + delta_pose[3]) != 0
        else:
            want_to_jump = actions.squeeze() > 0.5  # actions from beta distribution are in the range [0, 1]

        # check if the jump is requested and the cooldown is over
        self.jumping_now = want_to_jump & (self.jump_cooldown_buffer.eq(0))

        # update cooldown buffer
        self.jump_cooldown_buffer = torch.clamp(self.jump_cooldown_buffer - 1, min=0)

        if self.jumping_now.any():
            # set the jump cooldown buffer, starting when the jump is initiated, not when it ends
            self.jump_cooldown_buffer[self.jumping_now] = self.jump_cooldown_steps

    def apply_actions(self):
        """called each sim step"""

        if self.jumping_now.any():
            joint_pos = self._asset.data.joint_pos[self.jumping_now, 2].clone().unsqueeze(1)
            joint_vel = torch.zeros_like(joint_pos) + self.jump_velocity

            self._asset.write_joint_state_to_sim(
                position=joint_pos,
                velocity=joint_vel,
                joint_ids=[2],
                env_ids=torch.nonzero(self.jumping_now).squeeze(1),
            )

            self.jumping_now = torch.zeros_like(self.jumping_now, dtype=torch.bool)


class ArticulatedWrench2DAction(ActionTerm):
    """Action term for an rigid body that is attached to links preventing it from
    rolling and pitching. The action term applies a 2D wrench to the body in the body frame."""

    cfg: actions_cfg.ArticulatedWrench2DActionCfg
    _env: ManagerBasedRLEnv
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.ArticulatedWrench2DActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env
        self.max_velocity_forward = cfg.max_velocity_forward
        self.max_velocity_backward = cfg.max_velocity_backward
        self.max_vel_sideways = cfg.max_vel_sideways
        self.max_rot_vel = cfg.max_rotvel

        self.vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # unlimit joint ranges
        max_translate_xy = 1000.0
        max_translate_z = 1000.0
        max_rotate = 1e9
        limit_tensor = torch.tensor(
            [
                [-max_translate_xy, max_translate_xy],
                [-max_translate_xy, max_translate_xy],
                [-max_translate_z, max_translate_z],
                [-max_rotate, max_rotate],
            ],
        ).to(self.device)
        limit_tensor_envs = limit_tensor.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._asset.write_joint_limits_to_sim(limit_tensor_envs)

        # set link masses
        self._asset.data.default_mass[:, :-1] *= 0.0001

        # enable teleoperation
        if TELEOP:
            self.teleop_interface = Se3Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
            self.teleop_interface.add_callback("L", env.reset)
            print(self.teleop_interface)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 3 if not self.cfg.disable_translation else 1

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return torch.empty()

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    def process_actions(self, actions: torch.Tensor):
        """called each env step, i.e., once per {decimation} sim steps
        Actions from beta distribution are in the range [0, 1]"""
        # set joint effort targets

        if TELEOP:
            delta_pose, gripper_command = self.teleop_interface.advance()
            action_x = torch.zeros(self.env.num_envs).to(self.device) + delta_pose[0] * 4
            action_y = torch.zeros(self.env.num_envs).to(self.device) + delta_pose[1] * 4
            action_r_z = torch.zeros(self.env.num_envs).to(self.device) + delta_pose[2] * 4
            if self.cfg.disable_translation:
                action_x = torch.zeros(self.num_envs, 1, device=self.device)
                action_y = torch.zeros(self.num_envs, 1, device=self.device)

        elif self.cfg.disable_translation:
            action_x = torch.zeros(self.num_envs, 1, device=self.device)
            action_y = torch.zeros(self.num_envs, 1, device=self.device)
            action_r_z = self.scale_tensor(actions.squeeze(), (0, 0.5, 1), (-self.max_rot_vel, 0, self.max_rot_vel))

        else:
            ## scale action to max vel:
            ## ie scale [0, 1] to [-max_vel, max_vel]
            action_x = self.scale_tensor(
                actions[:, 0], (0, 0.5, 1), (-self.max_velocity_backward, 0, self.max_velocity_forward)
            )
            action_y = self.scale_tensor(actions[:, 1], (0, 0.5, 1), (-self.max_vel_sideways, 0, self.max_vel_sideways))
            action_r_z = self.scale_tensor(actions[:, 2], (0, 0.5, 1), (-self.max_rot_vel, 0, self.max_rot_vel))

            ## limit action to max vel:
            # vel_norm = torch.linalg.norm(actions[:, :2], dim=1)
            # above_max_vel = vel_norm > self.max_lin_vel
            # actions[above_max_vel, :2] = (
            #     actions[above_max_vel, :2] / vel_norm[above_max_vel].unsqueeze(1) * self.max_lin_vel
            # )

            # action_x = actions[:, 0]
            # action_y = actions[:, 1]
            # action_r_z = torch.clamp(actions[:, 2], -self.max_rot_vel, self.max_rot_vel)

        vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        vel_b[:, 0] = action_x.squeeze()
        vel_b[:, 1] = action_y.squeeze()
        rot_vel = action_r_z.squeeze()

        self.vel_b = vel_b
        self.rot_vel = rot_vel

    def apply_actions(self):
        """called each sim step"""

        robot_quat = self._asset.data.body_quat_w[:, -1, :]
        vel_w = math_utils.quat_apply_yaw(robot_quat, self.vel_b)

        self._asset.set_joint_velocity_target(vel_w[:, :2], joint_ids=[0, 1])
        self._asset.set_joint_velocity_target(self.rot_vel.unsqueeze(1), joint_ids=[3])

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "vel_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/action_vel"

                self.vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.vel_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- goal pose

        # - get robot pos
        robot_pos = self._asset.data.body_pos_w[:, -1, :].clone()
        robot_quat = self._asset.data.body_quat_w[:, -1, :]
        # increase z to visualize the vel
        robot_pos[:, 2] += 1.0

        # - get action vel
        vel = self.vel_b
        # vel_3d = torch.cat(self.vel_command, dim=1)
        vel_w = math_utils.quat_apply(robot_quat, self.vel_b)
        vel_x = vel_w[:, 0]
        vel_y = vel_w[:, 1]

        yaw_angle = torch.atan2(vel_y, vel_x)

        vel_dir_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_angle), torch.zeros_like(yaw_angle), yaw_angle
        )

        scales = torch.linalg.norm(vel, dim=1, keepdim=True)
        default_scale = torch.ones_like(scales) * 4
        scales_3d = torch.cat([scales * 2, default_scale, default_scale], dim=1)

        self.vel_visualizer.visualize(translations=robot_pos, orientations=vel_dir_quat, scales=scales_3d)

    def scale_tensor(
        self,
        input_tensor: torch.Tensor,
        input_range: tuple[float, float, float],
        output_range: tuple[float, float, float],
    ) -> torch.Tensor:
        """Scales the input tensor from the input range to the output range.
        Args:
            input_tensor: The input tensor to be scaled.
            input_range: The range of the input tensor.
            output_range: The range of the output tensor.
        """
        input_min, input_mid, input_max = input_range
        output_min, output_mid, output_max = output_range

        lower_scale = (output_mid - output_min) / (input_mid - input_min)
        upper_scale = (output_max - output_mid) / (input_max - input_mid)

        lower_offset = output_min - lower_scale * input_min
        upper_offset = output_mid - upper_scale * input_mid

        # Apply scaling in one step using where
        return torch.where(
            input_tensor <= input_mid,
            lower_scale * input_tensor + lower_offset,
            upper_scale * input_tensor + upper_offset,
        )
