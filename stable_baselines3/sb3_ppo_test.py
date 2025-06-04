import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from custom_intersection_env import CustomIntersectionEnv
from simple_intersection_env import SimpleIntersectionEnv


# ─────────────────────────────────────────────────────────────────────────────
# Create directories for model and plots
model_dir = "./stable_baselines3/ppo_cont/model/"
plots_dir = "./stable_baselines3/ppo_cont/plots/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom callback to track episode rewards & lengths
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.last_ep_count = 0

    def _on_step(self) -> bool:
        # Check for newly finished episodes in ep_info_buffer
        epi_buffer = self.model.ep_info_buffer
        if len(epi_buffer) > self.last_ep_count:
            for idx in range(self.last_ep_count, len(epi_buffer)):
                ep_info = epi_buffer[idx]
                if ep_info is not None:
                    self.rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])
            self.last_ep_count = len(epi_buffer)
        return True

    def plot_metrics(self):
        # Rewards per episode (scatter + line)
        if len(self.rewards) > 0:
            x = list(range(len(self.rewards)))
            y = self.rewards
            plt.figure(figsize=(8, 4))
            plt.scatter(x, y, s=10, alpha=0.5, color="tab:blue", label="Episode Reward")
            plt.plot(x, y, color="tab:blue", alpha=0.7, linewidth=1.0)
            plt.title("PPO‐Continuous: Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "rewards.png"))
            plt.close()

        # Episode lengths (scatter + line)
        if len(self.episode_lengths) > 0:
            x = list(range(len(self.episode_lengths)))
            y = self.episode_lengths
            plt.figure(figsize=(8, 4))
            plt.scatter(x, y, s=10, alpha=0.5, color="tab:green", label="Episode Length")
            plt.plot(x, y, color="tab:green", alpha=0.7, linewidth=1.0)
            plt.title("PPO‐Continuous: Episode Lengths")
            plt.xlabel("Episode")
            plt.ylabel("Length (# steps)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "episode_lengths.png"))
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-10, 10],
            "vy": [-10, 10],
        },
        "absolute": False,
        "clip": False,
        "normalize": False,
    },
    # ─────────────── Switch to continuous actions ───────────────
    "action": {
        "type": "ContinuousAction",
    },
    # ───────────────────────── Other settings ─────────────────────────
    "duration": 2,
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "destination": "o1",
    "initial_vehicle_count": 20,
    "spawn_probability": 0.8,
    "ego_spacing": 25,
    "initial_lane_id": None,
    "controlled_vehicles": 1,
    "duration": 15,
    "vehicles_density": 1.0,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "normalize_reward": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Register your custom environment
gym.envs.registration.register(
    id="simple-intersection-v0",
    entry_point="simple_intersection_env:SimpleIntersectionEnv",
)

# 1) Create the raw (unwrapped) env and wrap in Monitor (for ep_info_buffer)
raw_env = gym.make(
    "simple-intersection-v0", render_mode="rgb_array", config=CONFIG
)
monitored_env = Monitor(raw_env)

# 2) Vectorize
env = DummyVecEnv([lambda: monitored_env])

# 3) Create metrics callback
metrics_callback = MetricsCallback()

# ─────────────────────────────────────────────────────────────────────────────
# Configure PPO for ContinuousAction
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1000,           # on‐policy rollout length
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,          # you can set True for state‐dependent noise
    sde_sample_freq=-1,
    tensorboard_log="./ppo_cont_tb/",
    verbose=0,
    device="cpu",
)

# ─────────────────────────────────────────────────────────────────────────────
# Train for 100k timesteps, tracking episode‐level metrics
model.learn(
    total_timesteps=5000,
    callback=[ProgressBarCallback(), metrics_callback],
)

# Save plots after training
metrics_callback.plot_metrics()

# Save the trained model
model.save(os.path.join(model_dir, "ppo_continuous_intersection"))

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (human‐view rendering)
eval_env = gym.make("simple-intersection-v0", render_mode="human", config=CONFIG)
model = PPO.load(os.path.join(model_dir, "ppo_continuous_intersection"))

obs, _ = eval_env.reset()
episode_reward = 0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    episode_reward += reward
    eval_env.render()
    time.sleep(0.05)

    if done or truncated:
        crashed = info.get("crashed", False)
        arrived = info.get("arrived", False)
        print(f"Episode finished, total reward = {episode_reward}. crashed: {crashed}, arrived: {arrived}. Resetting…")
        episode_reward = 0
        obs, _ = eval_env.reset()
        time.sleep(1)

eval_env.close()
