"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.intrinsic_motivation import StyleInstructor

# Import extensions to set up environment tasks
import interactive_navigation.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

# import metra policy
from rsl_rl.modules.metra.actor import StochasticActor


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.episode_length_s = 3.0
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, extras = env.get_observations()
    timestep = 0

    # random policy:
    # env.action_space.shape,
    std_range = 1.0, 5.0
    num_random_envs = int(env.num_envs / 2)
    random_policy_stds = torch.linspace(std_range[0], std_range[1], num_random_envs, device=env.unwrapped.device)
    shuffle_idx = torch.randperm(num_random_envs)
    random_policy_stds = random_policy_stds[shuffle_idx]

    random_policy_means = torch.zeros(num_random_envs, env.num_actions, device=env.unwrapped.device)
    action_stds = random_policy_means + random_policy_stds.unsqueeze(1)
    random_policy_normal = torch.distributions.Normal(random_policy_means, action_stds)
    random_policy_uniform = torch.distributions.Uniform(-action_stds, action_stds)

    env_idx = torch.arange(env.num_envs, device=env.unwrapped.device)
    policy_action_mask = env_idx % 2 == 0
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device)

    # bad metra policy
    metra_obs = extras["observations"]["metra_policy"]
    bad_metra_policy_path = (
        "/home/rafael/Projects/MT/CRL/interactive_navigation/logs/rsl_rl/metra_anymal_test/anymal_loc_bad/model_2000.pt"
    )
    metra_state_dict = torch.load(bad_metra_policy_path, weights_only=True)["model_state_dict"]["policy"]
    # make skill tensor
    input_shape = metra_state_dict[list(metra_state_dict.keys())[1]].shape[1]
    raw_obs_shape = 0
    for _, tensor in metra_obs.items():
        raw_obs_shape += tensor.flatten(1).shape[-1]
    skill_dim = input_shape - raw_obs_shape
    skills = torch.eye(skill_dim, device=env.unwrapped.device)
    repeates = int(env.num_envs // skill_dim) + 1
    skill_tensor = skills.repeat(repeates, 1)[: env.num_envs]
    skill_dict = {"skill": skill_tensor}
    # load weights
    metra_obs |= skill_dict
    bad_metra_policy = StochasticActor(actor_obs_dict=metra_obs, num_actions=env.action_space.shape[1])
    bad_metra_policy.to(env.unwrapped.device)
    bad_metra_policy.load_state_dict(metra_state_dict)

    # set up the instructor
    instructor_obs = extras["observations"]["instructor"]
    style_instructor = StyleInstructor(input_shape=instructor_obs.shape[1])
    style_instructor.to(env.unwrapped.device)
    curr_date_str = os.popen("date +'%Y-%m-%d_%H-%M-%S'").read().strip()
    instructor_save_path = os.path.join(
        os.path.dirname(log_root_path), "style_guide", f"{curr_date_str}_style_instructor.pt"
    )
    os.makedirs(os.path.dirname(instructor_save_path), exist_ok=True)

    # simulate environment
    metra_ratio = 0.8
    count = 0
    current_best_loss = 1e6
    loss = []
    env.episode_length_buf = torch.randint(0, env.max_episode_length, (env.num_envs,), device=env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if count % 50 == 0:
                # alternate between normal and uniform
                choose_normal = torch.rand(num_random_envs, device=env.unwrapped.device) < 0.5
                choose_metra = torch.rand(num_random_envs, device=env.unwrapped.device) < metra_ratio
            count += 1

            # - agent stepping
            # policy action
            good_actions = policy(obs[policy_action_mask])
            # bad metra policy action
            bad_metra_actions = bad_metra_policy.act(extras["observations"]["metra_policy"] | skill_dict)[
                ~policy_action_mask
            ]
            # random actions
            random_actions_normal = random_policy_normal.sample()
            random_actions_uniform = random_policy_uniform.sample()
            random_actions = torch.where(choose_normal.unsqueeze(1), random_actions_normal, random_actions_uniform)
            # negative actions
            bad_actions = torch.where(choose_metra.unsqueeze(1), bad_metra_actions, random_actions)
            # combine actions
            actions[policy_action_mask] = good_actions
            actions[~policy_action_mask] = bad_actions

            # - env stepping
            obs, _, _, extras = env.step(actions)

        # instructor update
        instructor_obs = extras["observations"]["instructor"]
        positive_instructor_obs = instructor_obs[policy_action_mask]
        negative_instructor_obs = instructor_obs[~policy_action_mask]
        style_loss = style_instructor.update(positive_instructor_obs, negative_instructor_obs)
        loss.append(style_loss)
        if count % 50 == 0:
            current_loss = np.mean(loss)
            if current_loss < current_best_loss:
                current_best_loss = current_loss
                print(f"New best loss: {current_best_loss}")
                # save the model
                style_instructor.save_jit(instructor_save_path)

            else:
                print(f"Current loss: {current_loss}")
            loss = []
        # print(f"Style loss: {style_loss}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
