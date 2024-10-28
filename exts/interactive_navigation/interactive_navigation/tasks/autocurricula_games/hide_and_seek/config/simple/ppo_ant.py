"""Here we change the crl env to be compatible with ppo.
This consits of defining rewards, and changing the observations to a single group "policy",
which contains observations and goals"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

import interactive_navigation.tasks.autocurricula_games.hide_and_seek.mdp as mdp
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.ant_env_cfg import CrlAntEnvCfg, ObservationsCfg


@configclass
class RewardsCfg:
    """Rewards for PPO."""

    close_to_goal = RewTerm(func=mdp.close_to_goal, params={"command_name": "robot_goal"}, weight=1.0)

    at_goal = RewTerm(
        func=mdp.at_goal, params={"command_name": "robot_goal", "threshold": 0.34, "interpolate": False}, weight=100.0
    )

    move_to_goal = RewTerm(func=mdp.moving_towards_goal, params={"command_name": "robot_goal"}, weight=1.0)

    # episode_termination = RewTerm(
    #     func=mdp.is_terminated,
    #     weight=200.0,  #
    # )


@configclass
class PPO_ObservationsCfg:
    """single "policy" obs group"""

    @configclass
    class PPOPolicyObs(ObservationsCfg.PolicyCfg, ObservationsCfg.PolicyGoalCfg):
        """combine observations and goals"""

    policy: PPOPolicyObs = PPOPolicyObs()


@configclass
class PPOAntEnvCfg(CrlAntEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # change observations
        self.observations: PPO_ObservationsCfg = PPO_ObservationsCfg()
        # add rewards
        self.rewards: RewardsCfg = RewardsCfg()
