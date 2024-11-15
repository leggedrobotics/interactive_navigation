# train command:
# python scripts/rsl_rl/train.py --task Isaac-Games-HideAndSeek-Simple-D-v0 --num_envs 128 --headless --video --video_length 200 --video_interval 5000 --logger wandb --experiment_name move_boxes_to_center --log_project_name move_boxes_to_center
from interactive_navigation.tasks.autocurricula_games.hide_and_seek.articulation_env_cfg import N_BOXES

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoRelationalActorCriticCfg,
    RslRlPpoRecurrentActorCriticCfg,
)


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
class JumpeOnBoxesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 123
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "jump_on_boxes_ppo"
    run_name = f"jump_{N_BOXES}_box_N_step"
    wandb_project = "jump_on_boxes_ppo"
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
