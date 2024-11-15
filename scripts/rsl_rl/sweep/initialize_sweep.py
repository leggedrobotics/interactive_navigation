import wandb
import yaml
import os
import argparse
import json


def load_sweep_config(file_path) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Initialize a WandB sweep.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the WandB project.")
    parser.add_argument(
        "--entity_name",
        type=str,
        required=True,
        help="Name of the WandB entity/team. (e.g. https://wandb.ai/<entity_name>)",
    )
    args = parser.parse_args()

    # Get the path of this script
    sweep_path = os.path.dirname(os.path.abspath(__file__))
    sweep_config_path = os.path.join(sweep_path, "sweep.yaml")

    # Load the sweep configuration
    sweep_config = load_sweep_config(sweep_config_path)

    # Initialize the sweep and get the sweep ID
    sweep_id = wandb.sweep(sweep_config, project=args.project_name, entity=args.entity_name)
    print(f"Sweep '{args.project_name}' initialized with ID: {sweep_id}")

    # Path to the JSON file storing sweep names and IDs
    out_path = os.path.join(sweep_path, "sweep_ids.json")

    # Load existing sweep IDs if the file exists
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            sweep_ids = json.load(f)
    else:
        sweep_ids = {}

    # Update the dictionary with the new sweep name and ID
    sweep_ids[args.project_name] = sweep_id

    # Write the updated dictionary back to the JSON file
    with open(out_path, "w") as f:
        json.dump(sweep_ids, f, indent=4)


if __name__ == "__main__":
    main()
