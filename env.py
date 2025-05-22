import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import yaml
import sys

def __main__():
    # Read command line argument to get the config file
    if len(sys.argv) < 2:
        print("Usage: python env.py <config_file.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    env = gym.make(config["env"], config=config["config"])
    env.reset()