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


@configclass
class AnymalTestCrlRunnerCfg(RslCRlOnPolicyRunnerCfg):
    num_steps_per_env = 1000
    buffer_fill_steps = 1000
    num_learning_steps = 1
    update_actor_critic_simultaneously = True
    max_iterations = 250
    save_interval = 50
    experiment_name = "contrastive_RL_anymal"
    run_name = "crl_anymal_w"
    wandb_project = "crl_anymal_test"
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
        stack_N_critic_batches=4,
        actor_batch_size=256,
        replay_buffer_size_per_env=5000,
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