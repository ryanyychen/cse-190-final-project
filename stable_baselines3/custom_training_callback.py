import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class RewardTrackingCallback(BaseCallback):
    def __init__(self, tag, path_dir=".", verbose=0):
        super().__init__(verbose)
        self.all_rewards = []
        self.tag = tag
        self.path_dir = path_dir
        os.makedirs(path_dir, exist_ok=True)
        self.current_reward = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.all_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

    def _save_plot(self):
        if not self.all_rewards:
            return
        plt.figure()
        plt.plot(self.all_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.savefig(f"{self.path_dir}/{self.tag}.png")
        plt.close()

    def save_all_plot(self):
        self._save_plot()