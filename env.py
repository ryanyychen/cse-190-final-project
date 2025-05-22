import yaml
import highway_env
import gymnasium as gym

def create_env(config_filepath):
    """
    Create a gym environment based on the provided configuration file.

    Parameters:
    - config_filepath (str): Path to the YAML configuration file.

    Returns:
    - env: The created gym environment.
    """
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)

    if (config["render_mode"] == "none"):
        config["render_mode"] = None

    env = gym.make(config["env"], config=config["config"], render_mode=config["render_mode"])
    return env