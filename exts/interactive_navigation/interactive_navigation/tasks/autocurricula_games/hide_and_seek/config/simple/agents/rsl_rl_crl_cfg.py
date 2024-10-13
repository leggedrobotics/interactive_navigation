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
@configclass
class TestCrlRunnerCfg(RslCRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 100_000
    save_interval = 1000
    experiment_name = "contrastive_RL_test"
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
        mini_batch_size_and_num_inserts_per_sample=512,
        replay_buffer_size=1_000_000,
        num_critic_learning_epochs=2,
        num_actor_learning_epochs=2,
        entropy_coef=0.001,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        max_grad_norm=1.0,
    )
