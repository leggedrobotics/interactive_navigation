from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.utils import configclass, math as math_utils

if TYPE_CHECKING:
    from .actions_cfg import SimpleActionCfg


class SimpleAction(ActionTerm):
    """Simple action for the spherical agent to move around and grab obstacles."""

    cfg: SimpleActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SimpleActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_name = "robot"
        self.env = env

        self.max_force = 2.0  # Newtons
        self.max_torque = 1.0  # Newton-meters

        self._processed_actions = torch.zeros(3)

        # command buffers
        self.force_command = torch.zeros(env.num_envs, 3)
        self.torque_command = torch.zeros(env.num_envs, 3)

        # object pos buffers
        self.grabbed_asset_pos_body_frame = torch.zeros(env.num_envs, 3).to(env.device)
        self.grabbed_asset_quat_body_frame = torch.zeros(env.num_envs, 4).to(env.device)
        self.grabbed_asset_id = torch.zeros(env.num_envs).to(env.device) - 1

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 3 + 1  # x, y, yaw (continuous), grab (binary)

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return torch.empty()

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """

        # - movement x, y, yaw:
        # movement is implemented as a force in the x, y direction and a torque in the yaw direction
        xy_force_body = actions[:, :2] * self.max_force
        yaw_torque = actions[:, 2] * self.max_torque

        xy_force_body = torch.clamp(xy_force_body, -self.max_force, self.max_force)
        yaw_torque = torch.clamp(yaw_torque, -self.max_torque, self.max_torque)

        self.force_command = torch.cat([xy_force_body, torch.zeros(self.env.num_envs, 1).to(self.device)], dim=1)
        self.torque_command = torch.cat(
            [torch.zeros(self.env.num_envs, 2).to(self.device), yaw_torque.unsqueeze(1)], dim=1
        )

        # - grab:
        # when grab is activated (< 0), the closest object within a radius becomes movable for this agent
        self.is_grabbing = actions[:, 3] > -1000000
        # TODO facing direction

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """

        # - moving action
        # get forces
        # TODO put in process_actions

        self._asset.set_external_force_and_torque(
            forces=self.force_command.unsqueeze(1), torques=self.torque_command.unsqueeze(1)
        )

        # - grab actions:
        min_distance = 4.0  # meters

        # robot quaternion
        robot_quat = math_utils.yaw_quat(self.env.scene.rigid_objects["robot"].data.root_quat_w)

        # get positions
        robot_pos = self.env.scene.rigid_objects["robot"].data.root_pos_w

        # TODO: fix this, does not work properly

        asset_ids = ["asset_1", "asset_2", "asset_3"]
        for id_num, asset_id in enumerate(asset_ids):
            asset_pos = self.env.scene.rigid_objects[asset_id].data.root_pos_w

            # get distances
            dist_vec = asset_pos - robot_pos
            dist = torch.linalg.vector_norm(dist_vec, dim=1)
            # temporary action disabled
            grab_asset = (dist < min_distance) & self.is_grabbing

            # the boxes are kinematic bodies, meaning they cannot be pushed.
            # we move them by setting their position directly

            # if the agent was grabbing previously and its still grabing, we update the position of the grabbed asset.
            still_grabbing = (self.grabbed_asset_id == id_num) & grab_asset
            if still_grabbing.any():
                asset_pos_body_frame = self.grabbed_asset_pos_body_frame[still_grabbing]
                asset_quat_body_frame = self.grabbed_asset_quat_body_frame[still_grabbing]
                asset_pos_world_frame = (
                    math_utils.quat_apply(robot_quat[still_grabbing], asset_pos_body_frame) + robot_pos[still_grabbing]
                )
                asset_quat_world_frame = math_utils.quat_mul(robot_quat[still_grabbing], asset_quat_body_frame)
                asset_quat = self.env.scene.rigid_objects[asset_id].data.root_quat_w
                # fill pose tensor for all envs to write to sim
                asset_quat[still_grabbing] = asset_quat_world_frame
                asset_pos[still_grabbing] = asset_pos_world_frame
                asset_pose_world_frame = torch.cat([asset_pos, asset_quat], dim=1)
                self.env.scene.rigid_objects[asset_id].write_root_pose_to_sim(asset_pose_world_frame)

            # if we start grabbing now, we store the relative position of the asset to the robot and set grabbed_asset to True
            now_grabbing = grab_asset & ~still_grabbing
            if now_grabbing.any():
                # set the position of the grabbed asset in the body frame of the robot
                self.grabbed_asset_pos_body_frame[now_grabbing] = math_utils.quat_apply(
                    math_utils.quat_inv(robot_quat[now_grabbing]), dist_vec[now_grabbing]
                )
                self.grabbed_asset_quat_body_frame[now_grabbing] = math_utils.quat_mul(
                    math_utils.quat_inv(robot_quat[now_grabbing]),
                    self.env.scene.rigid_objects[asset_id].data.root_quat_w[now_grabbing],
                )
                self.grabbed_asset_id[now_grabbing] = id_num
