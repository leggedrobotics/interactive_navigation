from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.utils import configclass, math as math_utils
from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse


if TYPE_CHECKING:
    from .actions_cfg import SimpleActionCfg


class SimpleAction(ActionTerm):
    """Simple action for the spherical agent to move around and grab obstacles.

    - movement x, y, yaw: continuous action to move the agent in the x, y direction and yaw

    - grab: binary action to grab the closest object within a radius and within the field of view

    - climb up: binary action to make the agent climb up by jumping.
                This action is only available every N steps to prevent the agent from jumping continuously."""

    cfg: SimpleActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: SimpleActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_name = "robot"
        self.env = env

        # movement config
        self.max_force = cfg.max_force  # Newtons
        self.max_torque = cfg.max_toque  # Newton-meters
        self.max_lin_vel = cfg.max_lin_vel  # m/s

        # grab config
        self.min_distance = cfg.min_distance  # meters
        self.half_fov_rad = cfg.fov_deg * torch.pi / 360.0  # radians
        self.min_valid_z_diff_to_grab = 0.2
        self.grab_height = 0.05  # meters. if grabbed, the object is lifted by this height
        self.horizontal_force_threshold = 1.0  # Newtons. if the force is higher, the object is let go
        self.contact_cool_down = int(math.ceil(0.2 / self.env.step_dt))
        self.contact_cooldown_buffer = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device)

        # jump config
        self.jump_cooldown_steps = int(math.ceil(cfg.jump_cooldown_secs / self.env.step_dt))
        jump_height = 0.75  # meters
        angle_deg = 80.0  # angle from the horizontal plane to jump vector
        g = 9.81
        jump_vel_up = math.sqrt(2 * g * jump_height)
        jump_vel_forward = jump_vel_up / math.tan(angle_deg * math.pi / 180.0)
        self.jump_vel = torch.tensor([jump_vel_forward, 0.0, jump_vel_up]).to(self.env.device)

        self._processed_actions = torch.zeros(3)

        # asset names
        self.asset_names = [asset for asset in list(env.scene.rigid_objects.keys()) if "asset" in asset]

        # command buffers
        self.force_command = torch.zeros(env.num_envs, 3).to(env.device)
        self.torque_command = torch.zeros(env.num_envs, 3).to(env.device)

        # object pos buffers
        self.grabbed_asset_pos_body_frame = torch.zeros(env.num_envs, 3).to(env.device)
        self.grabbed_asset_quat_body_frame = torch.zeros(env.num_envs, 4).to(env.device)
        self.grabbed_asset_id = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device) - 1

        # jump cooldown
        self.jump_cooldown_buffer = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device)

        # teleop:
        self.use_teleop = cfg.use_teleop
        if self.use_teleop:
            self.teleop_interface = Se3Keyboard(pos_sensitivity=self.max_force, rot_sensitivity=self.max_torque)
            self.teleop_interface.add_callback("L", env.reset)
            print(self.teleop_interface)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 3 + 1 + 1  # x, y, yaw (continuous), grab (binary), climb up (binary)

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
        if self.use_teleop:
            delta_pose, gripper_command = self.teleop_interface.advance()
            self.force_command[:, :2] = torch.tensor(delta_pose[:2]).to(self.env.device)
            self.torque_command[:, 2] = (
                torch.tensor(delta_pose[2]).to(self.env.device) / self.max_force * self.max_torque
            )
        else:
            xy_force_body = actions[:, :2] * self.max_force
            yaw_torque = actions[:, 2] * self.max_torque

            xy_force_body = torch.clamp(xy_force_body, -self.max_force, self.max_force)
            yaw_torque = torch.clamp(yaw_torque, -self.max_torque, self.max_torque)
            yaw_torque *= 0

            self.force_command = torch.cat([xy_force_body, torch.zeros(self.env.num_envs, 1).to(self.device)], dim=1)
            self.torque_command = torch.cat(
                [torch.zeros(self.env.num_envs, 2).to(self.device), yaw_torque.unsqueeze(1)], dim=1
            )

        # - grab:
        # when grab is activated (< 0), the closest object within a radius becomes movable for this agent
        if self.use_teleop:
            self.is_grabbing = gripper_command
        else:
            self.is_grabbing = actions[:, 3] > 0
        self.contact_cooldown_buffer = torch.clamp(self.contact_cooldown_buffer - 1, min=0)

        # - climb up:
        # when climb up is activated (< 0), the agent jumps up by setting its velocity up and forward for one step
        if self.use_teleop:
            want_to_jump = (torch.tensor(delta_pose[3]).to(self.env.device) / self.max_force * self.max_torque) != 0
            jump_now = want_to_jump & (self.jump_cooldown_buffer.eq(0))
        else:
            jump_now = (actions[:, 4] > 0) & (self.jump_cooldown_buffer.eq(0))

        if jump_now.any():
            # set velocities
            all_robot_velocities = self.env.scene.rigid_objects[self.robot_name].data.root_lin_vel_w
            jump_robots_quat = self.env.scene.rigid_objects[self.robot_name].data.root_quat_w[jump_now]
            jump_vel_vec_world = math_utils.quat_apply_yaw(
                jump_robots_quat, self.jump_vel.unsqueeze(0).expand(int(jump_now.sum()), -1)
            )
            all_robot_velocities[jump_now] = jump_vel_vec_world
            zero_ang_vel = torch.zeros_like(all_robot_velocities)
            velocity = torch.cat([all_robot_velocities, zero_ang_vel], dim=1)
            self._asset.write_root_velocity_to_sim(velocity)
            # set cooldown
            self.jump_cooldown_buffer[jump_now] = self.jump_cooldown_steps
            self.jump_cooldown_buffer[~jump_now] = torch.clamp(self.jump_cooldown_buffer[~jump_now] - 1, min=0)
        else:
            self.jump_cooldown_buffer = torch.clamp(self.jump_cooldown_buffer - 1, min=0)

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """

        # - moving action

        # get forces

        # check if the robot is moving too fast
        robot_lin_vel_w = self.env.scene.rigid_objects[self.robot_name].data.root_lin_vel_w
        above_max_lin_vel = torch.linalg.vector_norm(robot_lin_vel_w[:, :2], dim=1) > self.max_lin_vel

        # if the robot is moving too fast, we set the force in the direction of the velocity to 0
        # by subtracting the projection of the force on the velocity from the force
        if above_max_lin_vel.any():
            # TODO define as torch jit function for performance
            robot_lin_vel_w = robot_lin_vel_w[above_max_lin_vel]
            robot_quat = self.env.scene.rigid_objects[self.robot_name].data.root_quat_w[above_max_lin_vel]
            robot_lin_vel_b = math_utils.quat_apply_yaw(math_utils.quat_inv(robot_quat), robot_lin_vel_w)

            xy_force = self.force_command[above_max_lin_vel, :2]
            # projection = (a.b / |b|^2) * b
            dot_products = torch.sum(xy_force * robot_lin_vel_b[:, :2], dim=1)
            norms_squared = torch.sum(robot_lin_vel_b[:, :2] ** 2, dim=1)
            eps = 1e-8  # small constant for numerical stability
            norms_squared = torch.clamp(norms_squared, min=eps)
            projection = (dot_products / norms_squared).unsqueeze(-1) * robot_lin_vel_b[:, :2]

            xy_force -= projection
            self.force_command[above_max_lin_vel, :2] = xy_force

        self._asset.set_external_force_and_torque(
            forces=self.force_command.unsqueeze(1), torques=self.torque_command.unsqueeze(1)
        )

        # - grab actions:
        # TODO grab distance should be to the surface of the asset, not its center
        # we only grab small objects, so we can assume the center is close to the surface

        # robot quaternion
        robot_quat_raw = self.env.scene.rigid_objects["robot"].data.root_quat_w
        robot_quat = math_utils.yaw_quat(robot_quat_raw)

        # get positions
        robot_pos = self.env.scene.rigid_objects["robot"].data.root_pos_w

        # (force the agent to not roll or pitch too much or at all)
        # tilt_error = math_utils.quat_error_magnitude(robot_quat, robot_quat_raw)
        # tilted_too_much = tilt_error > 0.75  # radians
        # robot_set_quat = torch.where(tilted_too_much, robot_quat, robot_quat_raw)
        robot_pose_toset = torch.cat([robot_pos, robot_quat], dim=1)
        self.env.scene.rigid_objects["robot"].write_root_pose_to_sim(robot_pose_toset)

        # check if the agent is grabbing an object
        grabbed_any = torch.zeros(self.env.num_envs, dtype=torch.bool).to(self.env.device)
        grabbed_new_ids = torch.zeros(self.env.num_envs, dtype=torch.long).to(self.env.device) - 1
        closest_asset_dist = torch.ones(self.env.num_envs).to(self.env.device) * self.min_distance * 2

        for id_num, asset_id in enumerate(self.asset_names):
            asset_pos = self.env.scene.rigid_objects[asset_id].data.root_pos_w

            # get distances
            dist_vec = asset_pos - robot_pos
            dist = torch.linalg.vector_norm(dist_vec, dim=1)

            # the boxes are kinematic bodies, meaning they cannot be pushed.
            # we move them by setting their position directly

            # if the agent was grabbing previously and its still grabbing, we update the position of the grabbed asset.
            # we increase the grabbing distance to avoid accidentally letting go of the asset
            potential_still_grabbing = (
                (dist < self.min_distance * 1.5) & self.is_grabbing & self.contact_cooldown_buffer.eq(0)
            )

            if potential_still_grabbing.any():
                # if still grabbing, we need to check if the agent is pushing the asset into a wall
                contact_sensor = self.env.scene.sensors["boxes_contact_forces"]
                contact_forces = contact_sensor.data.net_forces_w[potential_still_grabbing, id_num]
                horizontal_force = torch.linalg.vector_norm(contact_forces[:, :2], dim=1)
                too_much_force = horizontal_force > self.horizontal_force_threshold
                potential_grabbing_valid = torch.zeros_like(potential_still_grabbing, dtype=torch.bool)
                potential_grabbing_valid[potential_still_grabbing] = ~too_much_force

                # set cooldown for the asset if the force is too high
                start_force_cooldown = torch.zeros_like(self.contact_cooldown_buffer, dtype=torch.bool)
                start_force_cooldown[potential_still_grabbing] = too_much_force
                self.contact_cooldown_buffer[start_force_cooldown] = self.contact_cool_down + 1
                potential_still_grabbing = potential_grabbing_valid

            # check if the agent is grabbing and the asset is within the grabbing distance
            potential_asset_grab = (dist < self.min_distance) & self.is_grabbing
            # for assets in to be grabbed, check if the assets are within the field of view, not to far in z direction,
            # and the robot not just jumped
            if potential_asset_grab.any():
                asset_to_grab_pos_body_frame = math_utils.quat_apply_yaw(
                    math_utils.quat_inv(robot_quat[potential_asset_grab]), dist_vec[potential_asset_grab]
                )
                # x is forward, y is left
                in_fov = (
                    torch.abs(torch.atan2(asset_to_grab_pos_body_frame[:, 1], asset_to_grab_pos_body_frame[:, 0]))
                    < self.half_fov_rad
                )
                # check if asset is not below the robot, since this could be exploited by the agent
                valid_z_diff = torch.abs(dist_vec[potential_asset_grab][:, 2]) < self.min_valid_z_diff_to_grab
                # check if the agent is not just jumped
                not_jumped = self.jump_cooldown_buffer.eq(0)
                # check if the asset did not just collide:
                not_collided = self.contact_cooldown_buffer.eq(0)
                valid_grab = not_jumped & not_collided
                valid_grab[potential_asset_grab] &= in_fov & valid_z_diff
                potential_asset_grab = valid_grab

            # if the agent is still grabbing, we update the position of the grabbed asset
            still_grabbing = (self.grabbed_asset_id == id_num) & potential_still_grabbing
            if still_grabbing.any():

                # set the position of the grabbed asset in the body frame of the robot
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

            # if we start grabbing for the first time, we store the relative position of the asset to the robot and set grabbed_asset to True
            # if multiple assets are within the grabbing distance, we only grab the closest one
            closest_asset = dist < closest_asset_dist
            now_grabbing = potential_asset_grab & ~still_grabbing & self.grabbed_asset_id.eq(-1) & closest_asset
            if now_grabbing.any():

                closest_asset_dist[now_grabbing] = dist[now_grabbing]

                # set the position of the grabbed asset in the body frame of the robot
                self.grabbed_asset_pos_body_frame[now_grabbing] = math_utils.quat_apply(
                    math_utils.quat_inv(robot_quat[now_grabbing]), dist_vec[now_grabbing]
                ) + torch.tensor([0.0, 0.0, self.grab_height]).to(self.env.device)
                self.grabbed_asset_quat_body_frame[now_grabbing] = math_utils.quat_mul(
                    math_utils.quat_inv(robot_quat[now_grabbing]),
                    self.env.scene.rigid_objects[asset_id].data.root_quat_w[now_grabbing],
                )
                grabbed_new_ids[now_grabbing] = id_num

            grabbed_any = grabbed_any | potential_asset_grab | still_grabbing

        self.grabbed_asset_id[grabbed_new_ids >= 0] = grabbed_new_ids[grabbed_new_ids >= 0]
        self.grabbed_asset_id[~grabbed_any] = -1
