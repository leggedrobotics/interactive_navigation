from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.utils.timer import Timer, TIMER_CUMULATIVE
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.commands import GoalCommand
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp.utils import (
    get_robot_pos,
    get_robot_quat,
    get_robot_lin_vel_w,
    get_robot_rot_vel_w,
)


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


def any_box_close_to_step_reward(
    env: ManagerBasedRLEnv,
    robot_str: str,
    dist_sensor_1_str: str,
    dist_sensor_2_str: str,
    proximity_threshold: float,
    proximity_std: float,
    step_size_threshold: float,
) -> torch.Tensor:
    """Compute dense reward based on the proximity of any box to a step that can be jumped.
    Reward is given for closest box to the step on the same level as the robot.

    Args:
        env: The environment object.
        robot_str: Substring to match for the robot.
        dist_sensor_1_str: Substring to match for the first set of distance sensors.
        dist_sensor_2_str: Substring to match for the second set of distance sensors.
        proximity_threshold: Used in reward scaling; higher values shift the reward curve.
        proximity_std: Determines the steepness of the reward function.
        step_size_threshold: Minimum width of the step to be considered valid.

    Returns:
        torch.Tensor: Reward tensor for each environment.
    """
    # get z pos
    robot_pos = get_robot_pos(env.scene[robot_str])[..., 2]

    # Get sensor names matching the given substrings
    sensor_names_1 = sorted([name for name in env.scene.sensors if dist_sensor_1_str in name])
    sensor_names_2 = sorted([name for name in env.scene.sensors if dist_sensor_2_str in name])

    # Ensure we have the same number of sensors in both lists
    assert len(sensor_names_1) == len(sensor_names_2), "Sensor lists must be of equal length."

    min_dist_to_step = torch.full((env.num_envs,), float("inf"), device=env.device)

    for name1, name2 in zip(sensor_names_1, sensor_names_2):
        # Retrieve sensor data
        sensor1_data = env.scene.sensors[name1].data
        sensor2_data = env.scene.sensors[name2].data

        # Positions and hits in XY plane
        pos1 = sensor1_data.pos_w[..., :2].unsqueeze(1)
        z_pos1 = sensor1_data.pos_w[..., 2]
        hits1 = sensor1_data.ray_hits_w[..., :2]
        hits2 = sensor2_data.ray_hits_w[..., :2]

        z_diff = torch.abs(z_pos1 - robot_pos)
        distances1 = torch.linalg.vector_norm(hits1 - pos1, dim=2)
        diff_xy = torch.linalg.vector_norm(hits2 - hits1, dim=2)
        # a step is detected if the distance between the two sensors is greater than the threshold
        is_step = diff_xy > step_size_threshold
        same_z_level = z_diff < 0.5

        distances_to_steps = torch.where(is_step, distances1, torch.tensor(float("inf"), device=env.device))
        min_distances, _ = distances_to_steps.min(dim=1)
        min_distances[~same_z_level] = float("inf")

        min_dist_to_step = torch.min(min_dist_to_step, min_distances)

    # Compute the continuous reward based on the minimum distance to a step
    reward = 0.5 + 0.5 * torch.tanh((proximity_threshold - min_dist_to_step) / proximity_std)
    return reward


def closest_box_close_to_step_reward(
    env: ManagerBasedRLEnv,
    robot_str: str,
    dist_sensor_1_str: str,
    dist_sensor_2_str: str,
    proximity_threshold: float,
    proximity_std: float,
    step_size_threshold: float,
) -> torch.Tensor:
    """Compute dense reward based on the proximity of a box to a step that can be jumped.
    Reward is given only for the closest box to the robot on the same level as the robot.

    Args:
        env: The environment object.
        robot_str: Substring to match for the robot.
        dist_sensor_1_str: Substring to match for the first set of distance sensors.
        dist_sensor_2_str: Substring to match for the second set of distance sensors.
        proximity_threshold: Used in reward scaling; higher values shift the reward curve.
        proximity_std: Determines the steepness of the reward function.
        step_size_threshold: Minimum width of the step to be considered valid.

    Returns:
        torch.Tensor: Reward tensor for each environment.
    """
    robot_pos = get_robot_pos(env.scene[robot_str])

    # Get sensor names matching the given substrings
    sensor_names_1 = sorted([name for name in env.scene.sensors if dist_sensor_1_str in name])
    sensor_names_2 = sorted([name for name in env.scene.sensors if dist_sensor_2_str in name])

    # Ensure we have the same number of sensors in both lists
    assert len(sensor_names_1) == len(sensor_names_2), "Sensor lists must be of equal length."

    min_distances = torch.full((env.num_envs,), float("inf"), device=env.device)

    min_dist_box_robot = torch.full((env.num_envs,), float("inf"), device=env.device)
    for name1, name2 in zip(sensor_names_1, sensor_names_2):
        # Retrieve sensor data
        sensor1_data = env.scene.sensors[name1].data
        sensor2_data = env.scene.sensors[name2].data

        # Positions and hits in XY plane
        pos1 = sensor1_data.pos_w[..., :2]
        z_pos1 = sensor1_data.pos_w[..., 2]
        hits1 = sensor1_data.ray_hits_w[..., :2]
        hits2 = sensor2_data.ray_hits_w[..., :2]

        z_diff = torch.abs(z_pos1 - robot_pos[..., 2])
        distance_box_robot = torch.linalg.vector_norm(pos1 - robot_pos[..., :2], dim=1)
        distances1 = torch.linalg.vector_norm(hits1 - pos1.unsqueeze(1), dim=2)
        diff_xy = torch.linalg.vector_norm(hits2 - hits1, dim=2)

        # check if the box is closer to the robot than the previous closest box
        closest_to_robot = distance_box_robot < min_dist_box_robot
        min_dist_box_robot = torch.where(closest_to_robot, distance_box_robot, min_dist_box_robot)

        # a step is detected if the distance between the two sensors is greater than the threshold
        is_step = diff_xy > step_size_threshold
        same_z_level = z_diff < 0.5

        # find the min distance to a step
        distances_to_steps = torch.where(is_step, distances1, torch.tensor(float("inf"), device=env.device))
        # only consider min distance if the box is on the same level as the robot and closest to the robot
        min_distances[same_z_level & closest_to_robot] = distances_to_steps.min(dim=1)[0][
            same_z_level & closest_to_robot
        ]

    # Compute the continuous reward based on the minimum distance to a step
    reward = 0.5 + 0.5 * torch.tanh((proximity_threshold - min_distances) / proximity_std)
    return reward


class BoxMovingReward:

    def __init__(self):
        self.prev_box_pos = None

    def box_interaction(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Dense reward for moving box.
        Reward is the max position change of all boxes in one step."""

        asset_ids = [asset for asset in list(env.scene.rigid_objects.keys()) if "box" in asset]
        asset_poses = []
        for asset_id in asset_ids:
            asset_poses.append(env.scene.rigid_objects[asset_id].data.root_pos_w)
        positions = torch.stack(asset_poses, dim=1)[..., :2]

        if self.prev_box_pos is None:
            self.prev_box_pos = positions
            return torch.zeros(env.num_envs).to(env.device)

        positions_diff = positions - self.prev_box_pos

        max_diff = torch.max(torch.linalg.vector_norm(positions_diff, dim=-1), dim=-1)[0]
        # reward = -sum_distances / (len(asset_ids) * max(env.scene.terrain.cfg.terrain_generator.size))

        self.prev_box_pos = positions
        return max_diff


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
        robot_pos = get_robot_pos(env.scene["robot"])

        dist_to_boxes = torch.linalg.norm(self.boxes_positions_w - robot_pos, dim=-1)
        min_dist_to_box_per_env = torch.min(dist_to_boxes, dim=0).values

        half_max_dist = max(env.scene.terrain.cfg.terrain_generator.size) / 2

        reward = 1 - min_dist_to_box_per_env / half_max_dist

        return torch.clamp(reward, -1, 1)


class JumpReward:
    """Class for rewards related to the boxes."""

    def __init__(self) -> None:

        # previous z position history of the robot
        self.num_history_steps: int = 10
        self.step_delta_s: float = 0.2
        self.half_robot_heigh: float = 0.3
        self.step_delta: int = None  # type: ignore
        self.prev_height_buffer: torch.Tensor = None  # type: ignore
        self.max_height_reached: torch.Tensor = None  # type: ignore

    def _update_buffers(self, env: ManagerBasedRLEnv) -> None:
        """Update the buffers needed for the reward calculations."""

        # - robot height
        if self.prev_height_buffer is None:
            self.step_delta = int(self.step_delta_s / env.step_dt)
            self.prev_height_buffer = (
                torch.zeros((self.num_history_steps, env.num_envs)).to(env.device) + self.half_robot_heigh
            )
            self.max_height_reached = torch.zeros(env.num_envs).to(env.device) + self.prev_height_buffer[0]

        if env.common_step_counter % self.step_delta == 0:
            robot = env.scene["robot"]
            robot_height = get_robot_pos(robot)[:, 2]
            self.prev_height_buffer = torch.roll(self.prev_height_buffer, shifts=1, dims=0)
            self.prev_height_buffer[0] = robot_height

        if env.termination_manager.dones.any():
            # prev_heighs = self.prev_height_buffer.clone()
            # prev_heighs[:, env.termination_manager.dones] = self.half_robot_heigh
            # self.prev_height_buffer = prev_heighs
            self.prev_height_buffer[:, env.termination_manager.dones] = self.half_robot_heigh

            # max_height_reached = self.max_height_reached.clone()
            # max_height_reached[env.termination_manager.dones] = self.half_robot_heigh
            # self.max_height_reached = max_height_reached
            self.max_height_reached[env.termination_manager.dones] = self.half_robot_heigh

    def new_height_reached_reward(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Sparse reward function, given when the robot jumps and lands on a new max height level.
        A jump is considered successful if the robot is not moving vertically and the robot has reached a new maximum height
        """
        self._update_buffers(env)

        moving_vertically = (
            torch.sum(torch.abs(self.prev_height_buffer[1:4] - self.prev_height_buffer[:3]), dim=0) > 0.1
        )
        new_max_height_reached = self.prev_height_buffer[0] > (self.max_height_reached + 0.25)

        successfully_jumped = ~moving_vertically & new_max_height_reached

        self.max_height_reached = torch.where(successfully_jumped, self.prev_height_buffer[0], self.max_height_reached)

        reward = successfully_jumped.float()
        return reward

    def successful_jump_reward(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Sparse reward function that returns a reward when the robot jumps and lands on a higher level than it was before.
        A jump is considered successful if the robot is not moving vertically, the height difference between the
        initial and final height is greater than 0.3
        """
        self._update_buffers(env)

        moving_vertically = (
            torch.sum(torch.abs(self.prev_height_buffer[1:4] - self.prev_height_buffer[:3]), dim=0) > 0.1
        )
        changed_height = (
            torch.mean(self.prev_height_buffer[:4], dim=0) - torch.mean(self.prev_height_buffer[-4:], dim=0) > 0.2
        )
        successfully_jumped = ~moving_vertically & changed_height

        # if successfully_jumped, set prev_height_buffer to the current height
        if successfully_jumped.any():
            self.prev_height_buffer[:, successfully_jumped] = self.prev_height_buffer[0, successfully_jumped]
        reward = successfully_jumped.float()
        return reward


def outside_env(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Returns one if the robot is outside the environment, zero otherwise."""
    terrain_origins = env.scene.terrain.env_origins
    robot_pos = get_robot_pos(env.scene["robot"])

    diff = torch.linalg.vector_norm(terrain_origins - robot_pos, dim=-1)

    is_outside = diff > threshold

    return is_outside.float()


def action_penalty(env: ManagerBasedRLEnv, jump_penalty_factor: float = 2.0) -> torch.Tensor:
    """Penalize action."""
    action = env.action_manager.action

    max_vel = env.cfg.actions.wrench.max_velocity
    max_rotvel = env.cfg.actions.wrench.max_rotvel
    force_action = torch.clamp(torch.linalg.vector_norm(action[:, :2], dim=1), -max_vel, max_vel) / max_vel
    torque_action = torch.clamp(action[:, 2], -max_rotvel, max_rotvel) / max_rotvel
    jump_action = (action[:, 3] > 0).float()

    penalty = torch.abs(force_action) + torch.abs(torque_action) + jump_action

    return penalty


def close_to_goal(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Dense Reward for being close to the goal"""
    goal_command: GoalCommand = env.command_manager._terms[command_name]
    desired_goal = goal_command.goal
    current_goal = goal_command.command
    dist = torch.linalg.norm(desired_goal - current_goal, dim=-1)

    reward = torch.tanh((0.5 - dist) / 5) + 1
    return reward


def moving_towards_goal(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving towards the goal"""
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    robot_vel_w = get_robot_lin_vel_w(robot)
    robot_pos = get_robot_pos(robot)

    goal_command: GoalCommand = env.command_manager._terms[command_name]
    goal_pos_w = goal_command.goal_pos_w

    to_goal_vec = goal_pos_w - robot_pos
    to_goal_vec = to_goal_vec / torch.linalg.norm(to_goal_vec, dim=-1, keepdim=True)

    vel_to_goal = torch.sum(robot_vel_w * to_goal_vec, dim=-1)
    reward = torch.clamp(vel_to_goal, -1, 1)
    return reward


def at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    interpolate: bool = False,
) -> torch.Tensor:
    """Dense Reward for being at the goal"""
    goal_command: GoalCommand = env.command_manager._terms[command_name]
    desired_goal = goal_command.goal
    current_goal = goal_command.command
    dist = torch.linalg.norm(desired_goal - current_goal, dim=-1)
    at_goal = dist < threshold

    if interpolate:

        # linearly interpolate between 0 and 1 from threshold to 0
        linear_reward = (dist - threshold) / threshold
        reward = torch.where(at_goal, linear_reward, torch.zeros_like(linear_reward))
    else:
        reward = at_goal.float()

    return reward


def action_penalty_rigidbody(env: ManagerBasedRLEnv, jump_penalty_factor: float = 2.0) -> torch.Tensor:
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
    robot_pos = get_robot_pos(env.scene["robot"])

    height = torch.clamp(robot_pos[:, 2], height_range[0], height_range[1])
    reward = (height - height_range[0]) / (height_range[1] - height_range[0])
    return reward


##
# Style
##


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


##
#  Utility functions
##


def get_robot_pos(robot: Articulation | RigidObject) -> torch.Tensor:
    """Get the position of the robot."""
    if not isinstance(robot, (Articulation, RigidObject)):
        raise ValueError(f"Expected robot to be of type Articulation or RigidObject, got {type(robot)}")

    if isinstance(robot, Articulation):
        return robot.data.body_pos_w[:, -1, :]
    else:
        return robot.data.root_pos_w
