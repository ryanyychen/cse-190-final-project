import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
# Import your custom environment
from custom_intersection_env import CustomIntersectionEnv
import time

# Register your custom environment
gym.envs.registration.register(
    id='custom-intersection-v0',
    entry_point='custom_intersection_env:CustomIntersectionEnv', # replace with your path
)

# Now you can instantiate your new environment
# env = gym.make("custom-intersection-v0", render_mode='rgb_array')
env = gym.make("custom-intersection-v0", render_mode='rgb_array', config={
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of other vehicles to observe
        "features": ["presence", "x", "y", "vx", "vy"],  # Observe position and velocity
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",  # Keep simple, 5 discrete actions
    },
    "ego_spacing": 2,
    "initial_lane_id": None,
    "destination": "o3",
    "controlled_vehicles": 1,
    "policy_frequency": 5,
    "duration": 15,  # seconds
    "vehicles_density": 1.0,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # Intelligent traffic
    "idm_target_velocity": 10,
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "normalize_reward": False
})


# Train model again with new rewards
model = PPO("MlpPolicy", env, n_steps=100, verbose=0, device='cpu')
model.learn(total_timesteps=10000, callback=ProgressBarCallback())
model.save("./stable_baselines3_PPO/ppo_custom_intersection")

env.close()

env = gym.make("custom-intersection-v0", render_mode='human', config={
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of other vehicles to observe
        "features": ["presence", "x", "y", "vx", "vy"],  # Observe position and velocity
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",  # Keep simple, 5 discrete actions
    },
    "ego_spacing": 2,
    "initial_lane_id": None,
    "destination": "o3",
    "controlled_vehicles": 1,
    "policy_frequency": 5,
    "duration": 15,  # seconds
    "vehicles_density": 1.0,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # Intelligent traffic
    "idm_target_velocity": 10,
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "normalize_reward": False
})

model = PPO.load("./stable_baselines3_PPO/ppo_custom_intersection")

obs, _ = env.reset()
# ego = env.unwrapped.vehicle
# ego.route = ["ir0", "il1", "o1"]

# for key in env.unwrapped.road.network.graph:
#     print(key, env.unwrapped.road.network.graph[key])

# for start, end in env.unwrapped.road.network.graph:
#     print(f"Available road: {start} → {end}")

episode_reward = 0
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    episode_reward += reward

    # print(f"Step {step+1} - Reward: {reward:.3f}")

    env.render()
    time.sleep(0.05)

    if done or truncated:
        print(f"Episode finished, total reward of {episode_reward}, resetting...")
        episode_reward = 0
        obs, _ = env.reset()
        time.sleep(1)
