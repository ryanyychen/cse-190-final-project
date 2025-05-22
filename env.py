import gymnasium as gym
import yaml

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

    env = gym.make(config["env"], config=config["config"], render_mode=config["render_mode"])
    env.reset()
    return env