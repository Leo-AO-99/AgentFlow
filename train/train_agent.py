import yaml
import os
import subprocess
import sys
import argparse

def main():
    """
    Main function to parse YAML config, set environment variables,
    and run the training script.
    """
    parser = argparse.ArgumentParser(description="Run AgentFlow verl training.")
    parser.add_argument(
        "--config", default="train/config.yaml",
        help="Path to YAML config file (default: train/config.yaml)",
    )
    args, unknown_overrides = parser.parse_known_args()

    config_file_path = args.config

    # --- Parse YAML configuration ---
    print(f"Parsing YAML configuration from '{config_file_path}'...")
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_file_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # --- Set Environment Variables ---
    if 'env' in config:
        print("Setting environment variables...")
        for key, value in config['env'].items():
            # Use os.environ to set variables.
            # Convert non-string values to string.
            os.environ[key] = str(value)
            print(f"  Exported {key}={value}")

    # --- Construct Python Command Arguments ---
    # Start with the core command parts
    command = [sys.executable, "-m", "agentflow.verl"]

    # Get arguments from YAML and format them as 'key=value'
    if 'python_args' in config:
        print("Constructing Python command arguments...")
        for key, value in config['python_args'].items():
            # Support referencing environment variables in the YAML file
            # e.g., ${TRAIN_DATA_DIR}
            if isinstance(value, str):
                # Use os.path.expandvars to replace ${VAR} with its value
                expanded_value = os.path.expandvars(value)
                command.append(f"{key}={expanded_value}")
            else:
                command.append(f"{key}={value}")

    # Add any user-provided overrides to the command (e.g. data.train_batch_size=4)
    command.extend(unknown_overrides)
    
    # --- Execute the command ---
    print("\nStarting training script with the following command:")
    print(" ".join([str(item) for item in command]))
    print("-" * 50)

    try:
        # Use subprocess.run to execute the command.
        # env=os.environ passes all currently set environment variables.
        # check=True will raise an exception if the command returns a non-zero exit code.
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error: The training script failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found. "
              "Please make sure python is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()