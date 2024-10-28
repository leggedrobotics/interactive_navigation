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
"""Number of policy updates per step / number of value updates per step"""

beta_actor_value = 1 / 8
"""number of rollout steps per env per step/ number of value updates per step"""


# TODO check if this ratio considers the number of envs ie. is it per total transitions or per env transitions


"""
replaybuffer per env = replaybuffer total / num_envs = 5_000_000 / 4096 = 1220
crit batch size = mini_batch_size_and_num_inserts_per_sample * stack_N_critic_batches = 256 * 2 = 512
actor batch size = 1024

num_steps_sample_in_buffer = replaybuffer_per_env / num_steps_per_env = 1220 / 5 = 244

expected same sample used for:
 critic: crit_batch_size * num_critic_learning_steps / num_envs * num_steps_sample_in_buffer = 512 * 8 / 4096 * 244 = 244
 actor: actor_batch_size * num_actor_learning_steps / num_envs * num_steps_sample_in_buffer = 1024 * 4 / 4096 * 244 = 244

"""


@configclass
class TestCrlRunnerCfg(RslCRlOnPolicyRunnerCfg):
    num_steps_per_env = 1
    buffer_fill_steps = 500
    max_iterations = 1_000_000
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
        representation_dim=32,
        activation="elu",
    )

    algorithm = RslRlCrlAlgorithmCfg(
        mini_batch_size_and_num_inserts_per_sample=256,
        stack_N_critic_batches=2,
        actor_batch_size=1024,
        replay_buffer_size=2_000_000,
        num_critic_learning_steps=16,  # 2,
        num_actor_learning_steps=4,  # 1 / 4,
        entropy_coef=0.001,
        tau=0.005,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.95,
        max_grad_norm=1.0,
    )
