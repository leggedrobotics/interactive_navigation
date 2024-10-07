from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def dummy_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Placeholder reward function that returns a dummy reward."""
    asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "asset" in asset]

    asset_poses = []
    for asset_id in asset_ids:
        asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)

    positions = torch.stack(asset_poses)
    # positions is expected to be of shape (N, 2)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape (N, N, 2)
    dist_squared = (diff**2).sum(-1)  # shape (N, N)

    # Convert squared distances to actual distances
    dist = torch.sqrt(dist_squared)

    # Since dist[i, j] and dist[j, i] are the same, we can sum up one triangle and diagonal, then double it.
    # We avoid doubling the diagonal since it is always zero.
    triu_indices = torch.triu_indices(dist.size(0), dist.size(1), offset=1)
    sum_distances = dist[triu_indices[0], triu_indices[1]].sum(0)

    # TODO calculate a reward from this

    return torch.zeros(env.num_envs).to(env.device)


class BoxMovingReward:

    def __init__(self):
        self.prev_dist = None

    def obstacle_to_middle(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
        """Placeholder reward function that returns a dummy reward."""
        terrain_origins = env.scene.terrain.env_origins

        asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "box" in asset]
        asset_poses = []
        for asset_id in asset_ids:
            asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)
        positions = torch.stack(asset_poses)

        vec_to_center = terrain_origins.unsqueeze(0) - positions
        distances_to_center = torch.linalg.vector_norm(vec_to_center, dim=-1)

        if self.prev_dist is None:
            self.prev_dist = distances_to_center
            return torch.zeros(env.num_envs).to(env.device)

        at_goal = distances_to_center < threshold
        how_much_closer_than_before = self.prev_dist - distances_to_center
        moved_to_far = how_much_closer_than_before > 0.5

        reward = how_much_closer_than_before
        reward = torch.where(at_goal, torch.ones_like(reward), reward)
        reward = torch.where(moved_to_far, torch.zeros_like(reward), reward)

        # reward = -sum_distances / (len(asset_ids) * max(env.scene.terrain.cfg.terrain_generator.size))

        self.prev_dist = distances_to_center
        return torch.sum(reward, dim=0)


class CloseToBoxReward:
    """Class for rewards related to the boxes."""

    def __init__(self) -> None:
        # positions of the boxes in world coordinates
        self.boxes_positions_w: torch.Tensor = None  # type: ignore

    def _update_buffers(self, env: ManagerBasedRLEnv) -> None:
        """Update the buffers needed for the reward calculations."""

        # - box positions
        asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "box" in asset]
        asset_poses = []
        for asset_id in asset_ids:
            asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)

        self.boxes_positions_w = torch.stack(asset_poses)

    def close_to_box_reward(self, env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
        """Dense reward function that returns a reward when the robot is close to a box.
        Reward increases as the robot gets closer to the closest box from -1 to 1
        """
        self._update_buffers(env)

        robot = env.scene["robot"]
        robot_pos = robot.data.root_pos_w

        dist_to_boxes = torch.linalg.norm(self.boxes_positions_w - robot_pos, dim=-1)
        min_dist_to_box_per_env = torch.min(dist_to_boxes, dim=0).values

        half_max_dist = max(env.scene.terrain.cfg.terrain_generator.size) / 2

        reward = 1 - min_dist_to_box_per_env / half_max_dist

        return torch.clamp(reward, -1, 1)


class JumpReward:
    """Class for rewards related to the boxes."""

    def __init__(self) -> None:

        # previous z position history of the robot
        self.num_history_steps: int = 6
        self.step_delta_s: float = 0.1
        self.step_delta: int = None  # type: ignore
        self.prev_height_buffer: torch.Tensor = None  # type: ignore
        self.max_height_reached: torch.Tensor = None  # type: ignore

    def _update_buffers(self, env: ManagerBasedRLEnv) -> None:
        """Update the buffers needed for the reward calculations."""

        # - robot height
        if self.prev_height_buffer is None:
            self.step_delta = int(self.step_delta_s / env.step_dt)
            self.prev_height_buffer = (
                torch.zeros((self.num_history_steps, env.num_envs)).to(env.device)
                + env.scene["robot"].cfg.spawn.size[2] / 2
            )
            self.max_height_reached = torch.zeros(env.num_envs).to(env.device) + self.prev_height_buffer[0]

        if env.common_step_counter % self.step_delta == 0:
            robot = env.scene["robot"]
            robot_height = robot.data.root_pos_w[:, 2]
            self.prev_height_buffer = torch.roll(self.prev_height_buffer, shifts=1, dims=0)
            self.prev_height_buffer[0] = robot_height

    def successful_jump_reward(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Sparse reward function that returns a reward when the robot jumps and lands on a higher level.
        A jump is considered successful if the robot is not moving vertically, the height difference between the
        initial and final height is greater than 0.3 and the robot has reached a new maximum height
        """
        self._update_buffers(env)

        moving_vertically = (
            torch.sum(torch.abs(self.prev_height_buffer[1:] - self.prev_height_buffer[:-1]), dim=0) > 0.25
        )
        # changed_height = (
        #     torch.mean(self.prev_height_buffer[:4], dim=0) - torch.mean(self.prev_height_buffer[-4:], dim=0) > 0.3
        # )
        new_max_height_reached = self.prev_height_buffer[0] > (self.max_height_reached + 0.25)

        successfully_jumped = ~moving_vertically & new_max_height_reached

        self.max_height_reached = torch.where(successfully_jumped, self.prev_height_buffer[0], self.max_height_reached)

        reward = successfully_jumped.float()
        return reward


def outside_env(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Returns one if the robot is outside the environment, zero otherwise."""
    terrain_origins = env.scene.terrain.env_origins
    robot_pos = env.scene["robot"].data.root_pos_w

    diff = torch.linalg.vector_norm(terrain_origins - robot_pos, dim=-1)

    is_outside = diff > threshold

    return is_outside.float()


def action_penalty(env: ManagerBasedRLEnv, jump_penalty_factor: float = 2.0) -> torch.Tensor:
    """Penalize action."""
    action = env.action_manager.action

    max_force = env.cfg.actions.simple_wrench.max_force_forward
    max_torque = env.cfg.actions.simple_wrench.max_torque
    force_action = torch.clamp(torch.linalg.vector_norm(action[:, :2], dim=1), -max_force, max_force) / max_force
    torque_action = torch.clamp(action[:, 2], -max_torque, max_torque) / max_torque
    jump_action = (action[:, 3] > 0).float()

    penalty = torch.abs(force_action) + torch.abs(torque_action) + jump_action

    return penalty


def high_up(env: ManagerBasedRLEnv, height_range: tuple[float, float]) -> torch.Tensor:
    """Dense reward increaseing from 0 to 1 if the robot is within the heigh range."""
    robot_pos = env.scene["robot"].data.root_pos_w

    height = torch.clamp(robot_pos[:, 2], height_range[0], height_range[1])
    reward = (height - height_range[0]) / (height_range[1] - height_range[0])
    return reward
