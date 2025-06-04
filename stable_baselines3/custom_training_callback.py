import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class RewardTrackingCallback(BaseCallback):
    def __init__(self, tag, path_dir=".", verbose=0):
        super().__init__(verbose)
        self.all_rewards = []
        self.phase_rewards = []
        self.phase_index = 0
        self.tag = tag
        self.path_dir = path_dir
        os.makedirs(path_dir, exist_ok=True)
        self.current_reward = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.phase_rewards.append(self.current_reward)
            self.all_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

    def start_new_phase(self):
        """Call this between learn() calls to reset phase-specific reward logging."""
        self._save_phase_plot()
        self.phase_rewards = []
        self.phase_index += 1

    def _save_phase_plot(self):
        if not self.phase_rewards:
            return
        plt.figure()
        plt.plot(self.phase_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Training Rewards - Phase {self.phase_index + 1}")
        plt.savefig(f"{self.path_dir}/{self.tag}_phase{self.phase_index + 1}.png")
        plt.close()

    def save_all_plot(self):
        self._save_phase_plot()  # Save last phase
        plt.figure()
        plt.plot(self.all_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Combined Training Rewards Across All Phases")
        plt.savefig(f"{self.path_dir}/{self.tag}_combined.png")
        plt.close()

    

