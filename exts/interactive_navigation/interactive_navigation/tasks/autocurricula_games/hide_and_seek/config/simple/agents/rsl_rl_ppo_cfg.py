# train command:
# python scripts/rsl_rl/train.py --task Isaac-Games-HideAndSeek-Simple-D-v0 --num_envs 128 --headless --video --video_length 200 --video_interval 5000 --logger wandb --experiment_name move_boxes_to_center --log_project_name move_boxes_to_center

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoRelationalActorCriticCfg,
    RslRlPpoRecurrentActorCriticCfg,
    RslRlMetraAlgorithmCfg,
    SAC_MetraCfg,
)


##
# METRA
##


@configclass
class AntMetraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    num_transitions_per_episode = 4096
    max_iterations = 5_000
    save_interval = 2000
    experiment_name = "metra_ant_ppo_test"
    run_name = "metra_ant_ppo_test"
    wandb_project = "metra_ant_ppo"
    empirical_normalization = False
    policy = RslRlPpoRelationalActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    metra = RslRlMetraAlgorithmCfg(
        class_name="METRA",  # "METRA" with ppo or "METRA_SAC" with sac
        state_representation_args={
            "hidden_layers": [1024, 1024, 512],
            "activation": "elu",
        },
        batch_size=1024,
        instructor_reward_scaling=False,
        replay_buffer_size_per_env=100,
        replay_buffer_size_total=100000,
        num_metra_learning_epochs=1,
        num_sgd_steps_metra=250,
        skill_dim=2,
        lr=1e-4,
        lr_tau=1e-4,
        visualizer_interval=250,
        sac_hyperparameters=SAC_MetraCfg(  # only used if class_name is "METRA_SAC"
            gamma=0.99,
            alpha=0.2,
            polyak=0.005,
            lr=1e-4,
        ),
    )


##
# PPO
##


@configclass
class HideSeekPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 100_000
    save_interval = 1000
    experiment_name = "hide_and_seek_ppo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs * num_steps_per_env // num_mini_batches
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


"""Transformer-based relational actor-critic"""


@configclass
class HideSeekRelationalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 123
    num_steps_per_env = 64
    max_iterations = 100_000
    save_interval = 1000
    experiment_name = "transformer_ppo"
    run_name = "small_steps"
    wandb_project = "simple_articulation_ppo"
    empirical_normalization = False
    policy = RslRlPpoRelationalActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs * num_steps_per_env // num_mini_batches
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AntGoalCondPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 123
    num_steps_per_env = 64
    max_iterations = 100_000
    save_interval = 500
    experiment_name = "ant_goal_cond_ppo"
    run_name = "ant_goal_cond_ppo"
    wandb_project = "ant_goal_cond_ppo"
    empirical_normalization = False
    policy = RslRlPpoRelationalActorCriticCfg(
        class_name="GoalConditionedPPOActorCritic",
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs * num_steps_per_env // num_mini_batches
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


"""Recurrent actor-critic"""


@configclass
class RecurrentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 100_000
    save_interval = 1000
    experiment_name = "recurrent_ppo"
    empirical_normalization = False
    policy = RslRlPpoRecurrentActorCriticCfg(
        init_noise_std=5.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs * num_steps_per_env // num_mini_batches
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
