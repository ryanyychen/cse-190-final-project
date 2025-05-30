import os
import yaml
import numpy as np
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

class EnvWrapper:
    def __init__(self, env, config={}):
        self.env = env
        self.speed_factor = config.get("speed_factor", 0)
        self.steer_factor = config.get("steer_factor", 0)
        self.onroad_reward = config.get("onroad_reward", 0)
        self.offroad_penalty = config.get("offroad_penalty", 0)
        self.collision_penalty = config.get("collision_penalty", 0)
        self.offroad_terminal = config.get("offroad_terminal", False)
        self.collision_terminal = config.get("collision_terminal", False)
        self._override_reward()
    
    def _override_reward(self):
        original_reward_fn = self.env._reward

        def custom_reward(action):
            # Keep original reward logic
            reward = original_reward_fn(action)

            # Added custom reward logic
            vehicle = self.env.vehicle
            speed = vehicle.speed
            acc, steer = action

            # Penalize for speed over 5
            speed_penalty = self.speed_factor * max(0, speed - 5)
            reward -= speed_penalty

            # Penalize for excessive steering
            max_steer = self.env.config["vehicle"]["steering"]
            if abs(steer) * max_steer > 0.3:
                steer_penalty = self.steer_factor * abs(steer) * max_steer
                reward -= steer_penalty

            # Penalize if off the road, regardless of everything else
            if not vehicle.on_road:
                if self.offroad_penalty > 0:
                    reward -= self.offroad_penalty
            else:
                reward += self.onroad_reward
            
            # Penalize for collision
            if vehicle.crashed:
                if self.collision_penalty > 0:
                    reward -= self.collision_penalty
            
            # Reward for heading to destination
            # Extract destination
            destination = self.env.config["destination"]
            if vehicle.lane_index[0] == destination:
                reward += 5.0

            return reward
        
        self.env._reward = custom_reward
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, action):
        obs, obs_info, reward, done, info = self.env.step(action)
        if self.offroad_terminal and not self.env.vehicle.on_road:
            done = True
        if self.collision_terminal and self.env.vehicle.crashed:
            done = True
        return obs, obs_info, reward, done, info

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
        wrapped_env = EnvWrapper(
            env=env,
            config=config["wrapper_config"] if "wrapper_config" in config else {}
        )
        wrapped_env = GymnasiumRenderWrapper(wrapped_env) if render_mode else wrapped_env
        return wrapped_env
    except KeyError:
        raise ValueError(f"Environment {config['env']} is not registered. Please register it before creating the environment.")