import wandb
import argparse
import os
import json


def load_sweep_id(project_name, sweep_ids_path):
    """Load the sweep ID from the JSON file based on project name."""
    if os.path.exists(sweep_ids_path):
        with open(sweep_ids_path, "r") as f:
            sweep_ids = json.load(f)
        sweep_id = sweep_ids.get(project_name)
        if sweep_id:
            return sweep_id
        else:
            raise ValueError(f"No sweep ID found for project '{project_name}' in {sweep_ids_path}.")
    else:
        raise FileNotFoundError(f"Sweep IDs file not found at {sweep_ids_path}.")


def start_sweep_agent(sweep_id, agent_count=1):
    """Start the WandB sweep agent(s)."""
    wandb.agent(sweep_id, count=agent_count)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start WandB sweep agents using existing sweep ID.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the WandB project.")
    parser.add_argument(
        "--entity_name",
        type=str,
        required=True,
        help="Name of the WandB entity/team. (e.g. https://wandb.ai/<entity_name>)",
    )
    parser.add_argument("--agent_count", type=int, default=1, help="Number of sweep agents to start.")
    args = parser.parse_args()

    # Get the path of this script
    sweep_path = os.path.dirname(os.path.abspath(__file__))
    sweep_ids_path = os.path.join(sweep_path, "sweep_ids.json")

    # Load the sweep ID
    sweep_id = load_sweep_id(args.project_name, sweep_ids_path)
    print(f"Starting sweep agents for project '{args.project_name}' with sweep ID: {sweep_id}")

    # Start the sweep agents
    print(f"Starting {args.agent_count} sweep agent(s)...")
    wandb.agent(sweep_id, count=args.agent_count, entity=args.entity_name, project=args.project_name)


if __name__ == "__main__":
    main()
