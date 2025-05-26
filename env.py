import os
import yaml
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.parking_env import ParkingEnv
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.envs.intersection_env import IntersectionEnv 

class GymnasiumRenderWrapper:
    def __init__(self, env):
        self.env = env
        self.render_mode = "rgb_array"

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        obs, obs_info = self.env.reset(seed=seed)
        return obs

    def step(self, action):
        obs, obs_info, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        # ignore `mode` and just call original render
        return self.env.render()

def create_env(config_filepath, render_mode=None):
    """
    Create a gym environment based on the provided configuration file.

    Parameters:
    - config_filepath (str): Path to the YAML configuration file.
    - render_mode (str, optional): Mode for rendering the environment. Defaults to None. 'rgb_array' for video output
    - seed (int, optional): Seed for random number generation. Defaults to None.
    
    Returns:
    - env: The created gym environment.
    """

    ENV_CLASSES = {
        "merge-v0": MergeEnv,
        "two-way-v0": TwoWayEnv,
        "highway-v0": HighwayEnv,
        "parking-v0": ParkingEnv,
        "racetrack-v0": RacetrackEnv,
        "roundabout-v0": RoundaboutEnv,
        "intersection-v0": IntersectionEnv
    }

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Configuration file {config_filepath} does not exist.")

    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    assert "env" in config and "config" in config

    try:
        env_class = ENV_CLASSES[config["env"]]
        env = env_class(config=config["config"], render_mode=render_mode)
        wrapped_env = GymnasiumRenderWrapper(env) if render_mode else env
        return wrapped_env
    except KeyError:
        raise ValueError(f"Environment {config['env']} is not registered. Please register it before creating the environment.")