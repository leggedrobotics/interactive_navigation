from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from interactive_navigation.tasks.locomotion.box_climb.mdp.commands import GoalCommand

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


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


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def moving_towards_goal(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving towards the goal. This reward is computed as the dot product of the velocity of the robot
    and the unit vector pointing towards the goal."""
    robot = env.scene[robot_cfg.name]
    robot_vel_w = robot.data.root_lin_vel_w
    robot_pos = robot.data.root_pos_w

    goal_command: GoalCommand = env.command_manager._terms[command_name]
    goal_pos_w = goal_command.goal_pos_w

    to_goal_vec = goal_pos_w - robot_pos
    to_goal_vec = to_goal_vec / (torch.linalg.norm(to_goal_vec, dim=-1, keepdim=True) + 1e-6)

    vel_to_goal = torch.linalg.vecdot(robot_vel_w, to_goal_vec, dim=-1)
    reward = torch.clamp(vel_to_goal, -1, 1)
    return reward


def rotating_towards_goal_if_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dist_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for rotating towards the goal heading if the robot is within a distance threshold of the goal.
    The reward is computed as yaw velocity of the robot towards the goal."""
    robot = env.scene[robot_cfg.name]
    robot_yaw_vel_w = robot.data.root_ang_vel_w[:, 2]

    goal_command_generator: GoalCommand = env.command_manager._terms[command_name]
    heading_error_angle = goal_command_generator.heading_error_angle

    at_goal = torch.linalg.vector_norm(goal_command_generator.goal_pos_b, dim=1) < dist_threshold

    # If the angular error is small, we set it to zero
    heading_error_angle = torch.where(
        heading_error_angle.abs() < 0.1, torch.zeros_like(heading_error_angle), heading_error_angle
    )
    angvel_towards_goal = -heading_error_angle.sign() * robot_yaw_vel_w * at_goal.float()

    return angvel_towards_goal
