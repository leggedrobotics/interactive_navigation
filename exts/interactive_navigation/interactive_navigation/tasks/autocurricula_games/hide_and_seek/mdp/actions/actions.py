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

        self.max_velocity = 1.0
        self.max_angular_velocity = 1.0

        self._processed_actions = torch.zeros(3)

        self.force_command = torch.zeros(env.num_envs, 3)
        self.torque_command = torch.zeros(env.num_envs, 3)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 3

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
        xy_force_body = actions[:, :2] * self.max_velocity
        yaw_torque = actions[:, 2] * self.max_angular_velocity

        self.force_command = torch.cat([xy_force_body, torch.zeros(self.env.num_envs, 1).to(self.device)], dim=1)
        self.torque_command = torch.cat(
            [torch.zeros(self.env.num_envs, 2).to(self.device), yaw_torque.unsqueeze(1)], dim=1
        )

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """

        # get forces
        # TODO put in process_actions

        self._asset.set_external_force_and_torque(
            forces=self.force_command.unsqueeze(1), torques=self.torque_command.unsqueeze(1)
        )
