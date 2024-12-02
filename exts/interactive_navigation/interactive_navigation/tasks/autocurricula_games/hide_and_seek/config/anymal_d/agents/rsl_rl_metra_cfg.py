from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoRelationalActorCriticCfg,
    RslRlPpoRecurrentActorCriticCfg,
    RslRlMetraAlgorithmCfg,
    SAC_MetraCfg,
    StateReprCfg,
)


##
# METRA
##
@configclass
class AnymalMetraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25  # -1
    num_transitions_per_episode = None
    max_iterations = 20_000
    save_interval = 2000
    experiment_name = "metra_ppo_anymal_test"
    run_name = "metra_ppo_anymal_test"
    wandb_project = "metra_ppo_anymal"
    empirical_normalization = True
    policy = RslRlPpoRelationalActorCriticCfg(
        init_noise_std=1.0,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=15,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    metra = RslRlMetraAlgorithmCfg(
        class_name="METRA_PPO",  # METRA_PPO or METRA_SAC
        state_representation_args=StateReprCfg(
            hidden_layers=[1024, 1024, 512],
            activation="elu",
            layer_norm=False,
        ),
        batch_size=256,
        replay_buffer_size_per_env=100,
        replay_buffer_size_total=1_000_00,
        num_metra_learning_epochs=1,
        num_sgd_steps_metra=10,
        skill_dim=2,
        lr=5e-3,
        lr_tau=1e-4,
        visualizer_interval=250,
        # sac_hyperparameters=SAC_MetraCfg(
        #     gamma=0.999,
        #     alpha=0.22,
        #     polyak=0.005,
        #     lr=1e-4,
        # ),
        non_metra_reward_scale=0.5,
    )
