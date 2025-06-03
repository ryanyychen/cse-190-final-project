import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# Import your custom environment
from custom_intersection_env import CustomIntersectionEnv
import time

# 1. Change the “action” config to a continuous action type.
#    For highway‐env, that usually means something like "ContinuousAction".
CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy"],
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
    # ────────────────────────────────────────────────
    # THIS IS THE ONLY LINE YOU NEED TO EDIT FOR CONTINUOUS ACTIONS:
    "action": {
        "type": "ContinuousAction",  # ← was "DiscreteMetaAction" before
    },
    # ────────────────────────────────────────────────
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "destination": "o3",
    "initial_vehicle_count": 20,
    "spawn_probability": 0.95,
    "ego_spacing": 25,
    "initial_lane_id": None,
    "controlled_vehicles": 1,
    "duration": 15,
    "vehicles_density": 1.0,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "normalize_reward": False
}

# Register your custom environment exactly as before
gym.envs.registration.register(
    id='custom-intersection-v0',
    entry_point='custom_intersection_env:CustomIntersectionEnv',
)

# Create and wrap the env in a DummyVecEnv
env = gym.make("custom-intersection-v0", render_mode='rgb_array', config=CONFIG)
env = DummyVecEnv([lambda: env])


# 2. Import SAC instead of PPO; remove PPO‐only hyperparameters
#    Here’s a sensible SAC configuration to start with:
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    # buffer_size=100_000,       # Size of the replay buffer
    batch_size=256,            # Minibatch size sampled from replay buffer
    tau=0.005,                 # Target smoothing coefficient (for soft update)
    gamma=0.95,
    train_freq=1,              # How often to sample from the buffer
    gradient_steps=1,          # How many gradient steps per rollout
    ent_coef="auto",           # Let SAC automatically tune the entropy coefficient
    target_entropy="auto",     # Same for target entropy
    learning_starts=100,     # Number of steps before SAC starts updating
    use_sde=False,             # You can switch to True if you want state‐dependent noise
    tensorboard_log="./stable_baselines3/sac/sac_intersection_tensorboard/",
    verbose=0,
    device='cpu'
)

# 3. Train for as many timesteps as you like
model.learn(
    total_timesteps=500,
    callback=ProgressBarCallback()
)
model.save("./stable_baselines3/sac/sac_custom_intersection")

env.close()

# ───────────────────────────────────────────────────────────────────────
# Evaluation: load the SAC model and run in “human” render mode
# ───────────────────────────────────────────────────────────────────────

# Recreate the environment in human mode
env = gym.make("custom-intersection-v0", render_mode='human', config=CONFIG)
model = SAC.load("./stable_baselines3/sac/sac_custom_intersection")

obs, _ = env.reset()
episode_reward = 0

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
    time.sleep(0.05)

    if done or truncated:
        print(f"Episode finished, total reward = {episode_reward}. Resetting…")
        episode_reward = 0
        obs, _ = env.reset()
        time.sleep(1)

env.close()
