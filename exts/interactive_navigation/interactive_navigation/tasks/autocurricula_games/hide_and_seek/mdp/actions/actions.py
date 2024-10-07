# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import math

TELEOP = True

if TELEOP:
    from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse, Se3Gamepad
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
        self.jump_height = 0.75  # meters

        self.jump_cooldown_buffer = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device)
        self.is_jumping = torch.zeros(self.env.num_envs, dtype=torch.bool).to(self.env.device)
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
        return 1  # force x, y, torque z

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

        delta_pose, gripper_command = self.teleop_interface.advance()

        want_to_jump = (torch.zeros_like(actions.squeeze()) + delta_pose[3]) != 0
        self.want_to_jump = want_to_jump

        jump_now = want_to_jump & (self.jump_cooldown_buffer.eq(0)) & (~self.is_jumping)

        if jump_now.any():
            # set the jump flag and store the current height
            self.is_jumping[jump_now] = True
            self.bottom_height[jump_now] = self._asset.data.body_pos_w[jump_now, -1, 2]

            # set the jump cooldown buffer, starting when the jump is initiated, not when it ends
            self.jump_cooldown_buffer[jump_now] = self.jump_cooldown_steps
            self.jump_cooldown_buffer[~jump_now] = torch.clamp(self.jump_cooldown_buffer[~jump_now] - 1, min=0)
        else:
            self.jump_cooldown_buffer = torch.clamp(self.jump_cooldown_buffer - 1, min=0)

    def apply_actions(self):
        """called each sim step"""
        current_height = self._asset.data.body_pos_w[:, -1, 2]
        at_top = current_height - self.bottom_height > self.jump_height
        self.is_jumping[at_top] = False

        # if jumping, move up to the desired height
        if self.is_jumping.any():
            # reset effort limit
            self._asset.write_joint_effort_limit_to_sim(
                limits=torch.ones(self.is_jumping.sum().item(), 1).to(self.device) * 100.0,
                joint_ids=[2],
                env_ids=self.is_jumping.nonzero().squeeze(1),
            )

            # set the joint velocity target
            self._asset.set_joint_position_target(
                target=torch.ones(self.is_jumping.sum().item(), 1).to(self.device),
                joint_ids=[2],
                env_ids=self.is_jumping.nonzero().squeeze(1),
            )

        # if not jumping, relax the joint by setting the effort limit to 0
        if (~self.is_jumping).any():
            self._asset.write_joint_effort_limit_to_sim(
                limits=torch.zeros((~self.is_jumping).sum().item(), 1).to(self.device),
                joint_ids=[2],
                env_ids=(~self.is_jumping).nonzero().squeeze(1),
            )


class ArticulatedWrench2DAction(ActionTerm):
    """Action term for an rigid body that is attached to links preventing it from
    rolling and pitching. The action term applies a 2D wrench to the body in the body frame."""

    cfg: actions_cfg.ArticulatedWrench2DActionCfg
    _env: ManagerBasedRLEnv
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.ArticulatedWrench2DActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env
        self.max_lin_vel = 3.0

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
        return 3  # force x, y, torque z

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

        delta_pose, gripper_command = self.teleop_interface.advance()

        action_x = torch.zeros(self.env.num_envs) + delta_pose[0] * 1000
        action_y = torch.zeros(self.env.num_envs) + delta_pose[1] * 1000
        action_r_z = torch.zeros(self.env.num_envs) + delta_pose[2] * 1000

        # self._asset.set_joint_velocity_target(action_x, joint_ids=[0])  # self._joint_ids)
        # self._asset.set_joint_velocity_target(action_y, joint_ids=[1])
        # self._asset.set_joint_velocity_target(action_r_z, joint_ids=[3])

        # robot_quat_yaw = math_utils.yaw_quat(asset.data.body_quat_w[:, -1, :])

        force_b = torch.zeros(self.num_envs, 3, device=self.device)
        force_b[:, 0] = action_x.squeeze()
        force_b[:, 1] = action_y.squeeze()
        torque = torch.zeros(self.num_envs, 3, device=self.device)
        torque[:, 2] = action_r_z.squeeze()

        self.force_b = force_b
        self.torque = torque

    def apply_actions(self):
        """called each sim step"""
        # robot_quat = self._asset.data.body_quat_w[:, -1, :]
        # force_w = math_utils.quat_rotate(robot_quat, self.force_b)

        # targets = torch.cat([force_w[:, :2], self.torque[:, 2:]], dim=1)

        # check if the robot is moving too fast
        robot_lin_vel_w = self._asset.data.body_lin_vel_w[:, -1, :]
        above_max_lin_vel = torch.linalg.vector_norm(robot_lin_vel_w[:, :2], dim=1) > self.max_lin_vel

        # if the robot is moving too fast, we set the force in the direction of the velocity to 0
        # by subtracting the projection of the force on the velocity from the force
        if above_max_lin_vel.any():
            # TODO define as torch jit function for performance
            robot_lin_vel_w = robot_lin_vel_w[above_max_lin_vel]
            robot_quat = self._asset.data.body_quat_w[above_max_lin_vel, -1, :]
            robot_lin_vel_b = math_utils.quat_apply_yaw(math_utils.quat_inv(robot_quat), robot_lin_vel_w)

            xy_force = self.force_b[above_max_lin_vel, :2]
            # projection = (a.b / |b|^2) * b
            dot_products = torch.sum(xy_force * robot_lin_vel_b[:, :2], dim=1)
            norms_squared = torch.sum(robot_lin_vel_b[:, :2] ** 2, dim=1)
            eps = 1e-8  # small constant for numerical stability
            norms_squared = torch.clamp(norms_squared, min=eps)
            projection = (dot_products / norms_squared).unsqueeze(-1) * robot_lin_vel_b[:, :2]
            xy_force -= projection
            self.force_b[above_max_lin_vel, :2] = xy_force

        self._asset.set_external_force_and_torque(
            forces=self.force_b.unsqueeze(1), torques=self.torque.unsqueeze(1), body_ids=[-1]
        )

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
        robot_pos = self._asset.data.body_pos_w[:, -1, :].clone()
        robot_quat = self._asset.data.body_quat_w[:, -1, :]
        # increase z to visualize the force
        robot_pos[:, 2] += 1.0

        # - get action force
        force = self.force_b
        # force_3d = torch.cat(self.force_command, dim=1)
        force_w = math_utils.quat_apply(robot_quat, self.force_b)
        force_x = force_w[:, 0]
        force_y = force_w[:, 1]

        yaw_angle = torch.atan2(force_y, force_x)

        force_dir_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_angle), torch.zeros_like(yaw_angle), yaw_angle
        )

        scales = torch.linalg.norm(force, dim=1, keepdim=True) / 250
        default_scale = torch.ones_like(scales) * 4
        scales_3d = torch.cat([scales * 2, default_scale, default_scale], dim=1)

        self.force_visualizer.visualize(translations=robot_pos, orientations=force_dir_quat, scales=scales_3d)
