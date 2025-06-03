import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# Import your custom environment
from custom_intersection_env import CustomIntersectionEnv
import time

CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of other vehicles to observe
        "features": ["presence", "x", "y", "vx", "vy"],  # Observe position and velocity
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-10, 10],
            "vy": [-10, 10]
        },
        "absolute": False,
        "clip": False,
        "normalize": False
    },
    "action": {
        "type": "DiscreteMetaAction",  # Keep simple, 5 discrete actions
    },
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "destination": "o3",
    "initial_vehicle_count": 20,
    "spawn_probability": 0.8,
    "ego_spacing": 25,
    "initial_lane_id": None,
    "controlled_vehicles": 1,
    "duration": 15,  # seconds
    "vehicles_density": 1.0,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "normalize_reward": False
}

# Register your custom environment
gym.envs.registration.register(
    id='custom-intersection-v0',
    entry_point='custom_intersection_env:CustomIntersectionEnv',
)

# Create and wrap the environment
env = gym.make("custom-intersection-v0", render_mode='rgb_array', config=CONFIG)
env = DummyVecEnv([lambda: env])

# Configure PPO with better parameters
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,  # Increased batch size
    batch_size=64,
    n_epochs=10,  # Number of epochs when optimizing the surrogate loss
    learning_rate=3e-4,  # Learning rate
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Factor for trade-off of bias vs variance for GAE
    clip_range=0.2,  # Clipping parameter for PPO
    clip_range_vf=None,  # Clipping parameter for value function
    normalize_advantage=True,
    ent_coef=0.01,  # Entropy coefficient for exploration
    vf_coef=0.5,  # Value function coefficient
    max_grad_norm=0.5,  # Maximum norm for gradient clipping
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    tensorboard_log="./stable_baselines3/ppo/ppo_intersection_tensorboard/",
    verbose=0,
    device='cpu'
)

# Train for longer
model.learn(
    total_timesteps=1000,  # Much longer training
    callback=ProgressBarCallback()
)
model.save("./stable_baselines3/ppo/ppo_custom_intersection")

env.close()

# Evaluation
env = gym.make("custom-intersection-v0", render_mode='human', config=CONFIG)
model = PPO.load("./stable_baselines3/ppo/ppo_custom_intersection")

obs, _ = env.reset()
episode_reward = 0

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
    time.sleep(0.05)

    if done or truncated:
        print(f"Episode finished, total reward of {episode_reward}, resetting...")
        episode_reward = 0
        obs, _ = env.reset()
        time.sleep(1)
