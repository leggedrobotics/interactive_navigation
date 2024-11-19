## Wandb Sweeps
How to run sweeps for your isaac lab extension.
Wandb sweep documentation: https://docs.wandb.ai/guides/sweeps


#### General Procedure:
1. Define sweep configuration in `sweep.yaml`
2. Initialize sweep once in `initialize_sweepl.py` 
3. Run sweep in parallel by running `sweep.py`multiple times (on the cluster)

#### Instructions
In the `interactive_navigation/scripts/rsl_rl/sweep` directory, there are multiple files to configure, initialize and run sweeps.

##### Sweep configuration
In `interactive_navigation/scripts/rsl_rl/sweep/sweep.yaml` you can define the sweep, i.e., which parameters to optimize.
Detailed instructions can be found on the wandb sweep documentation: https://docs.wandb.ai/guides/sweeps/sweep-config-keys/#distribution-options-for-random-and-bayesian-search


**Important:**
1. Define the program path such that it points to the correct `train.py` file on the cluster.
2. The first command should be `/isaac-sim/python.sh` and not just `python`
3. Arguments need to be passed with an equal sign, i.e., `--num_envs=4096`
4. The hyper-parameters to be optimized are added with `${args_no_hyphens}`



**Example**
```yaml
program: interactive_navigation/scripts/rsl_rl/train.py
description: "Sweep over hyper parameters"
method: bayes  # or 'grid', 'random'
metric:
  name: Metric/success_rate  # wandb metric to optimize
  goal: maximize  # 'minimize' or 'maximize' if optimizing a loss
parameters:
# network parameters
  agent.policy.layer_dim:
    values: [32, 64, 128, 256]
  agent.algorithm.gamma:
    min: 0.99
    max: 0.999
    distribution: log_uniform_values
# reward weights
  env.rewards.goal_reached.weight:
    min: 50.0
    max: 500.0
    distribution: uniform

command:
  - /isaac-sim/python.sh
  - ${program}
  - --task=Isaac-MyTask-v0
  - --num_envs=4096
  - --logger=wandb
  - --headless
  - --video
  - --video_length=250
  - --video_interval=10000
  - ${args_no_hyphens}

```

##### Initialize Sweep
In  `interactive_navigation/scripts/rsl_rl/sweep/initialize_sweep.py`, you initialize the sweep **ONCE**.
```bash
python interactive_navigation/scripts/rsl_rl/sweep/initialize_sweep.py --project_name your_sweep_name --entity_name your_wandb_name
```

`--project_name` can be arbitrary. `--entity_name` is your wandb entity name. You can find it in the wandb web address: `https://wandb.ai/entity_name/project_name` 

This will write the sweep id into `interactive_navigation/scripts/rsl_rl/sweep/sweep_ids.json`.

```json
{
    "your_sweep_name": "nkbj7529",
    "your_other_sweep_name": "5j6ftpjh"
}
```
If you reinitialize a sweep with the same project name it will overwrite the sweep id. This is necessary if you change the `sweep.yaml`file.

##### Run Sweep

Once the sweep id is written, you can start the sweep by running.
```bash
python interactive_navigation/scripts/rsl_rl/sweep/sweep.py --project_name your_sweep_name --entity_name your_wandb_name
```

This command can be run on multiple machines to parallelize the sweep. Note, this only works if the sweep id is the same.
Set `--agent_count` to set the maximum number of runs per individual run. Typically, you run this on the cluster. To do so, you would submit multiple jobs with the exact same arguments.

I.e., repeat
```bash
./docker/cluster/cluster_interface.sh job --project_name your_sweep_name --entity_name your_wandb_name --agent_count 10
```
as many times as you whish.

**Note**: to run the sweep on the cluster, you need to change the cluster python executable in `.env.cluster`, i.e.:
```bash
# CLUSTER_PYTHON_EXECUTABLE=extension_name/scripts/rsl_rl/train.py
CLUSTER_PYTHON_EXECUTABLE=extension_name/scripts/rsl_rl/sweep/sweep.py
```

**Remember**: If you change the sweep config, you need to repeat all the steps.
