import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from custom_intersection_env import CustomIntersectionEnv

# ─────────────────────────────────────────────────────────────────────────────
# Create directories for model and plots
model_dir = "./stable_baselines3/sac/model/"
plots_dir = "./stable_baselines3/sac/plots/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom callback to track metrics for SAC
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Episode‐level metrics (populated at the end of each rollout)
        self.rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        # Training‐loss metrics (populated each time a gradient step is logged)
        self.losses = []

    def _on_step(self) -> bool:
        # Called at every call to `env.step()` (and every training update for off‐policy algs).
        # Check if SB3 has logged a new "train/loss"
        if "train/loss" in self.model.logger.name_to_value:
            loss_val = float(self.model.logger.name_to_value["train/loss"])
            self.losses.append(loss_val)
        return True

    def _on_rollout_end(self) -> None:
        # SAC is off‐policy, so `_on_rollout_end()` fires once after each call to `collect_rollouts()`.
        # By then, `ep_info_buffer` should hold one new entry if an episode ended during the rollout.
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if ep_info is not None:
                self.rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                self.episode_count += 1

    def plot_metrics(self):
        # 1) Plot Episode‐Total Rewards (scatter + line)
        if len(self.rewards) > 0:
            x = list(range(len(self.rewards)))
            y = self.rewards
            plt.figure(figsize=(10, 5))
            plt.scatter(x, y, alpha=0.5, s=10, color="tab:blue", label="Episode Reward")
            plt.plot(x, y, alpha=0.7, linewidth=1.0, color="tab:blue")
            plt.title("Training Rewards (per episode)")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "rewards.png"))
            plt.close()
        else:
            print("⚠️  No training rewards recorded; 'rewards.png' will be empty.")

        # 2) Plot Episode Lengths (scatter + line)
        if len(self.episode_lengths) > 0:
            x = list(range(len(self.episode_lengths)))
            y = self.episode_lengths
            plt.figure(figsize=(10, 5))
            plt.scatter(x, y, alpha=0.5, s=10, color="tab:green", label="Episode Length")
            plt.plot(x, y, alpha=0.7, linewidth=1.0, color="tab:green")
            plt.title("Episode Lengths (per episode)")
            plt.xlabel("Episode")
            plt.ylabel("Length (# steps)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "episode_lengths.png"))
            plt.close()
        else:
            print("⚠️  No episode lengths recorded; 'episode_lengths.png' will be empty.")

        # 3) Plot Training Loss (scatter + line)
        if len(self.losses) > 0:
            x = list(range(len(self.losses)))
            y = self.losses
            plt.figure(figsize=(10, 5))
            plt.scatter(x, y, alpha=0.5, s=10, color="tab:orange", label="Training Loss")
            plt.plot(x, y, alpha=0.7, linewidth=1.0, color="tab:orange")
            plt.title("Training Loss (per gradient step)")
            plt.xlabel("Gradient Update Index")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "losses.png"))
            plt.close()
        else:
            print("⚠️  No training losses recorded; 'losses.png' will be empty.")


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
    "action": {
        "type": "ContinuousAction",
    },
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "destination": "o3",
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
    id="custom-intersection-v0",
    entry_point="custom_intersection_env:CustomIntersectionEnv",
)

# 1) Create the raw (unwrapped) environment
raw_env = gym.make("custom-intersection-v0", render_mode="rgb_array", config=CONFIG)

# 2) Wrap in a Monitor so ep_info_buffer is populated
monitored_env = Monitor(raw_env)

# 3) Vectorize
env = DummyVecEnv([lambda: monitored_env])

# 4) Create the metrics callback
metrics_callback = MetricsCallback()

# ─────────────────────────────────────────────────────────────────────────────
# Configure SAC
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=1000,   # sufficiently large replay buffer
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    target_entropy="auto",
    learning_starts=1_000,
    use_sde=False,
    verbose=0,
    device="cpu",
)

# ─────────────────────────────────────────────────────────────────────────────
# Train for 100k timesteps (adjust as needed), tracking metrics
model.learn(
    total_timesteps=1000,
    callback=[ProgressBarCallback(), metrics_callback],
)

# Once training is done, plot and save all metrics
metrics_callback.plot_metrics()

# Save the trained SAC model
model.save(os.path.join(model_dir, "sac_custom_intersection"))

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (unchanged)
eval_env = gym.make("custom-intersection-v0", render_mode="human", config=CONFIG)
model = SAC.load(os.path.join(model_dir, "sac_custom_intersection"))

obs, _ = eval_env.reset()
episode_reward = 0

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    episode_reward += reward
    eval_env.render()
    time.sleep(0.05)

    if done or truncated:
        print(f"Episode finished, total reward = {episode_reward}. Resetting…")
        episode_reward = 0
        obs, _ = eval_env.reset()
        time.sleep(1)

eval_env.close()
