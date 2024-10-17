# train command:
# python scripts/rsl_rl/train.py --task Isaac-Games-HideAndSeek-Simple-D-v0 --num_envs 128 --headless --video --video_length 200 --video_interval 5000 --logger wandb --experiment_name move_boxes_to_center --log_project_name move_boxes_to_center

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl.crl_cfg import (
    RslRlGoalConditionedActorCfg,
    RslRlContrastiveCriticCfg,
    RslCRlOnPolicyRunnerCfg,
    RslRlCrlAlgorithmCfg,
)


##
# Test setup for contrastive RL
##

beta_policy_value = 1 / 2
"""Number of rollouts per step / number of value updates per step"""

beta_actor_value = 1 / 8
"""number of actor updates per step/ number of value updates per step"""


@configclass
class TestCrlRunnerCfg(RslCRlOnPolicyRunnerCfg):
    num_steps_per_env = 1
    buffer_fill_steps = 5000
    max_iterations = 100_000
    save_interval = 1000
    experiment_name = "contrastive_RL_test"
    run_name = "crl_test"
    wandb_project = "crl_test"
    empirical_normalization = False
    policy = RslRlGoalConditionedActorCfg(
        init_noise_std=1.0,
        activation="elu",
    )
    critic = RslRlContrastiveCriticCfg(
        representation_dim=128,
        activation="elu",
    )

    algorithm = RslRlCrlAlgorithmCfg(
        mini_batch_size_and_num_inserts_per_sample=256,
        stack_N_critic_batches=16,
        replay_buffer_size=5_000_000,
        num_critic_learning_steps=2,
        num_actor_learning_steps=1 / 4,
        entropy_coef=0.001,
        tau=0.001,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        max_grad_norm=1.0,
    )
