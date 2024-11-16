## Wandb Sweeps

In the `interactive_navigation/scripts/rsl_rl/sweep` directory, there are multiple files to configure, initialize and run sweeps.

In `interactive_navigation/scripts/rsl_rl/sweep/sweep.yaml` you can define the sweep, i.e., which parameters to optimize.

In  `interactive_navigation/scripts/rsl_rl/sweep/initialize_sweep.py`, you initialize the sweep **ONCE**.
```bash
python interactive_navigation/scripts/rsl_rl/sweep/initialize_sweep.py --project_name your_sweep_name
```
 This will writhe the sweep id into `interactive_navigation/scripts/rsl_rl/sweep/sweep_ids.json`.

Once the sweep id is written, you can start the sweep by running.
```bash
python interactive_navigation/scripts/rsl_rl/sweep/sweep.py --project_name your_sweep_name
```

This command can be run on multiple machines to parallelize the sweep. Note, this only works if the sweep id is the same.
If you have a big GPU, you can set `--project_name` to run multiple agents in parallel on one machine. Note, multiple GPUs are not supported

Note: to run the sweep on the cluster, you need to initialize the run with a sweep configuration yaml file `/isaac-sim/python.sh` instead of `python` in the command:

```yaml
command:
  - /isaac-sim/python.sh
  - ${program}
  # reset
  ```
  And this has to be done **before** initializing the run. This means you cannot run the same sweep on the cluster and on your local machine.