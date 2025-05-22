import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import yaml
import sys

def create_env(config_filepath):
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)

    env = gym.make(config["env"], config=config["config"])
    env.reset()
    return env