import os
import yaml
import numpy as np
from envs.merge_env import MergeEnv
from envs.two_way_env import TwoWayEnv
from envs.highway_env import HighwayEnv
from envs.parking_env import ParkingEnv
from envs.racetrack_env import RacetrackEnv
from envs.roundabout_env import RoundaboutEnv
from envs.intersection_env import IntersectionEnv 

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
        self.progress_reward = config.get("progress_reward", 0)
        self.offroad_penalty = config.get("offroad_penalty", 0)
        self.collision_penalty = config.get("collision_penalty", 0)
        self.wrongexit_penalty = config.get("wrongexit_penalty", 0)
        self.offroad_terminal = config.get("offroad_terminal", False)
        self.collision_terminal = config.get("collision_terminal", False)
        self.smoothness_reward = config.get("smoothness_reward", 0)
        self.consistency_reward = config.get("consistency_reward", 0)
        self.min_speed_reward = config.get("min_speed_reward", 0)
        self.max_speed_penalty = config.get("max_speed_penalty", 0)
        self.heading_alignment_reward = config.get("heading_alignment_reward", 0)
        self.distance_to_goal_reward = config.get("distance_to_goal_reward", 0)
        self.lane_keeping_reward = config.get("lane_keeping_reward", 0)
        self.turn_reward = config.get("turn_reward", 0)
        self.approach_reward = config.get("approach_reward", 0)
        self.exit_reward = config.get("exit_reward", 0)
        self.min_entropy = config.get("min_entropy", 0.05)
        self.max_entropy = config.get("max_entropy", 0.2)
        self.entropy_decay = config.get("entropy_decay", 0.999)
        self.last_action = None
        self.last_speed = None
        self.last_heading = None
        self.last_position = None
        self.last_lane = None
        self.entropy_coef = self.max_entropy
        self._override_reward()
    
    def _override_reward(self):
        original_reward_fn = self.env._reward

        def custom_reward(action):
            # Keep original reward logic
            reward = original_reward_fn(action)

            # If has_arrived was awarded, check if vehicle arriving at destination
            if reward == 50:
                # Lane Index structure: [current_road_id, destination_road_id, lane_index]
                if self.env.vehicle.lane_index[1] != self.env.config["destination"]:
                    reward = -self.wrongexit_penalty
                    return reward
                else:
                    reward += self.exit_reward
                    return reward

            # Get vehicle state
            vehicle = self.env.vehicle
            speed = vehicle.speed
            acc, steer = action
            heading = vehicle.heading
            position = vehicle.position
            current_lane = vehicle.lane_index[0]
            target_lane = vehicle.lane_index[1]

            # Calculate smoothness reward
            if self.last_action is not None:
                last_acc, last_steer = self.last_action
                acc_change = abs(acc - last_acc)
                steer_change = abs(steer - last_steer)
                smoothness = 1.0 - (acc_change + steer_change) / 2.0
                reward += self.smoothness_reward * smoothness

            # Calculate consistency reward
            if self.last_speed is not None and self.last_heading is not None:
                speed_consistency = 1.0 - abs(speed - self.last_speed) / max(speed, self.last_speed)
                heading_consistency = 1.0 - abs(heading - self.last_heading) / np.pi
                consistency = (speed_consistency + heading_consistency) / 2.0
                reward += self.consistency_reward * consistency

            # Speed-based rewards
            if speed > 0.5:  # Minimum speed threshold
                reward += self.min_speed_reward
            if speed > 5.0:  # Maximum safe speed
                reward -= self.max_speed_penalty * (speed - 5.0)

            # Penalize for speed over limit
            speed_penalty = self.speed_factor * max(0, speed - 5)
            reward -= speed_penalty

            # Penalize for excessive steering
            max_steer = self.env.config["vehicle"]["steering"]
            if abs(steer) * max_steer > 0.3:
                steer_penalty = self.steer_factor * abs(steer) * max_steer
                reward -= steer_penalty

            # Destination-based rewards
            dest = self.env.config["destination"]
            if dest:
                # Get current position relative to intersection center
                rel_x = position[0]
                rel_y = position[1]
                
                # Calculate desired heading based on current position and destination
                if dest == "o1":  # Straight
                    if rel_x < 0:  # Coming from left
                        target_heading = 0
                    else:  # Coming from right
                        target_heading = np.pi
                elif dest == "o2":  # Left
                    if rel_x < 0:  # Coming from left
                        target_heading = np.pi/2
                    else:  # Coming from right
                        target_heading = -np.pi/2
                elif dest == "o3":  # Right
                    if rel_x < 0:  # Coming from left
                        target_heading = -np.pi/2
                    else:  # Coming from right
                        target_heading = np.pi/2

                # Calculate heading alignment reward
                heading_diff = abs(heading - target_heading)
                heading_alignment = 1.0 - heading_diff / np.pi
                reward += self.heading_alignment_reward * heading_alignment

                # Reward for being in the correct lane for the destination
                if dest == "o1" and "ir" in current_lane and "il" in target_lane:
                    reward += self.progress_reward
                elif dest == "o2":  # Left turn
                    # Check if we're in the intersection and making a left turn
                    if "ir" in current_lane:
                        # Calculate if we're actually turning left based on heading
                        if rel_x < 0:  # Coming from left
                            if heading > 0 and heading < np.pi/2:  # Turning left
                                reward += self.turn_reward * 3  # Triple reward for correct turn
                                # Additional reward for being in the correct lane
                                if "il" in target_lane and target_lane != current_lane:
                                    reward += self.progress_reward * 2
                        else:  # Coming from right
                            if heading < 0 and heading > -np.pi/2:  # Turning left
                                reward += self.turn_reward * 3  # Triple reward for correct turn
                                # Additional reward for being in the correct lane
                                if "il" in target_lane and target_lane != current_lane:
                                    reward += self.progress_reward * 2
                        
                        # Penalize going straight when should turn left
                        if abs(heading) < 0.1:  # Going straight
                            reward -= self.turn_reward
                elif dest == "o3" and "ir" in current_lane and "il" in target_lane:
                    if "r" in target_lane:  # Right turn lane
                        reward += self.progress_reward
                        reward += self.turn_reward

                # Reward for approaching intersection correctly
                if "ir" in current_lane:
                    if dest == "o2":  # Approaching for left turn
                        # Check if we're in the correct lane for left turn
                        if rel_x < 0:  # Coming from left
                            if heading > 0 and heading < np.pi/2:  # Turning left
                                reward += self.approach_reward * 3  # Triple reward for correct approach
                                # Additional reward for being in the correct lane
                                if "il" in target_lane and target_lane != current_lane:
                                    reward += self.progress_reward * 2
                        else:  # Coming from right
                            if heading < 0 and heading > -np.pi/2:  # Turning left
                                reward += self.approach_reward * 3  # Triple reward for correct approach
                                # Additional reward for being in the correct lane
                                if "il" in target_lane and target_lane != current_lane:
                                    reward += self.progress_reward * 2
                    elif dest == "o3" and "r" in target_lane:  # Approaching for right turn
                        reward += self.approach_reward
                    elif dest == "o1" and "il" in target_lane:  # Approaching for straight
                        reward += self.approach_reward

            # Lane keeping reward
            if vehicle.on_road:
                lane_center = vehicle.lane.local_coordinates(vehicle.position)[1]
                lane_width = vehicle.lane.width
                lane_keeping = 1.0 - abs(lane_center) / (lane_width / 2)
                reward += self.lane_keeping_reward * lane_keeping

            # Penalize if off the road
            if not vehicle.on_road:
                if self.offroad_penalty > 0:
                    reward -= self.offroad_penalty
            else:
                reward += self.onroad_reward
            
            # Penalize for collision
            if vehicle.crashed:
                if self.collision_penalty > 0:
                    reward -= self.collision_penalty

            # Update last state
            self.last_action = action
            self.last_speed = speed
            self.last_heading = heading
            self.last_position = position
            self.last_lane = current_lane

            # Update entropy coefficient - keep it higher for longer
            self.entropy_coef = max(self.min_entropy, min(self.max_entropy, self.entropy_coef * self.entropy_decay))

            return reward
        
        self.env._reward = custom_reward

    def _get_target_heading(self):
        """Calculate the target heading based on the destination and current position."""
        if not self.env.config["destination"]:
            return 0
        
        dest = self.env.config["destination"]
        position = self.env.vehicle.position
        rel_x = position[0]
        
        if dest == "o1":  # Straight
            if rel_x < 0:  # Coming from left
                return 0
            else:  # Coming from right
                return np.pi
        elif dest == "o2":  # Left
            if rel_x < 0:  # Coming from left
                return np.pi/2
            else:  # Coming from right
                return -np.pi/2
        elif dest == "o3":  # Right
            if rel_x < 0:  # Coming from left
                return -np.pi/2
            else:  # Coming from right
                return np.pi/2
        return 0

    def _get_goal_position(self):
        """Get the approximate goal position based on the destination."""
        if not self.env.config["destination"]:
            return np.array([0, 0])
        
        dest = self.env.config["destination"]
        if dest == "o1":  # Straight
            return np.array([100, 0])
        elif dest == "o2":  # Left
            return np.array([0, 100])
        elif dest == "o3":  # Right
            return np.array([0, -100])
        return np.array([0, 0])
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, *args, **kwargs):
        self.last_action = None
        self.last_speed = None
        self.last_heading = None
        self.last_position = None
        self.last_lane = None
        self.entropy_coef = self.max_entropy
        return self.env.reset(*args, **kwargs)
    
    def step(self, action):
        obs, obs_info, reward, done, info = self.env.step(action)
        if self.offroad_terminal and not self.env.vehicle.on_road:
            done = True
            info["terminated_due_to_offroad"] = True
        if self.collision_terminal and self.env.vehicle.crashed:
            done = True
            info["terminated_due_to_collision"] = True
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