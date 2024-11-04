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
one epoch = one iteration of the main loop
    replaybuffer_size_per_env = 1000

    num_learning_samples_critic = num_learning_steps * num_critic_steps_per_update * mini_batch_size_and_num_inserts_per_sample * stack_N_critic_batches
    = 10 * 2 * 256 * 1 = 5120

    num_learning_samples_actor = num_learning_steps * num_actor_steps_per_update * actor_batch_size 
    = 10 * 2 * 256 = 5120

    num_new_samples_per_epoch = num_envs * num_steps_per_env = 4096 * 20 = 81_920

    ratio = num_new_samples_per_epoch / num_learning_samples_critic = 81_920 / 5120 = 16
    
    same_sample_in_buffer_for_n_steps = replaybuffer_size_per_env / num_steps_per_env = 1000 / 20 = 50
    
    expected_same_sample_reused_n_times = num_learning_samples_critic / (num_envs * replaybuffer_size_per_env) * same_sample_in_buffer_for_n_steps
    = 5120 / (4096 * 1000) * 50 = 0.0625 = 1/16
"""


@configclass
class TestCrlRunnerCfg(RslCRlOnPolicyRunnerCfg):
    num_steps_per_env = 1000
    buffer_fill_steps = 1000
    num_learning_steps = 1
    update_actor_critic_simultaneously = True
    max_iterations = 1_220_000
    save_interval = 50
    experiment_name = "contrastive_RL_ant"
    run_name = "crl_ant"
    wandb_project = "crl_RB_test"
    empirical_normalization = False
    policy = RslRlGoalConditionedActorCfg(
        class_name="GoalConditionedGaussianPolicy",
        init_noise_std=1.0,
        activation="elu",
    )
    critic = RslRlContrastiveCriticCfg(
        representation_dim=64,
        activation="elu",
    )

    algorithm = RslRlCrlAlgorithmCfg(
        mini_batch_size_and_num_inserts_per_sample=256,
        stack_N_critic_batches=8,
        actor_batch_size=256,
        replay_buffer_size_per_env=1000,
        num_critic_learning_steps_per_update=1,
        num_actor_learning_steps_per_update=1,
        log_sum_exp_regularization_coef=0.05,
        entropy_coef=0.001,
        tau=1.0,  #
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        use_target_entropy=True,  # None
        info_nce_type="forward",
        gamma=0.99,
        max_grad_norm=1.0,
    )
