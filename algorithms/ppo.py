import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
from gym_recorder import Recorder


class PolicyValueNetwork(nn.Module):
    """
    Combined policy and value network for PPO.
    Outputs:
      - action mean and log_std for continuous action distribution
      - state value estimate
    """
    def __init__(self, state_size, hidden_size, action_size):
        super(PolicyValueNetwork, self).__init__()
        # Shared base layers
        self.base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        # Actor head: outputs (action_size) means and (action_size) log_stds
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Linear(hidden_size, action_size)
        # Critic head: outputs a single scalar value
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize log_std bias to small negative value (so initial std ~ 0.1)
        nn.init.constant_(self.actor_log_std.bias, -4.0)

    def forward(self, state):
        """
        Given a batch of states, returns:
          - mean (batch_size x action_size)
          - log_std (batch_size x action_size)
          - value  (batch_size x 1)
        """
        x = self.base(state)
        mean = self.actor_mean(x)
        log_std = torch.clamp(self.actor_log_std(x), min=-20, max=2)
        value = self.critic(x)
        return mean, log_std, value


class PPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        learning_rate=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        k_epochs=4,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        update_timestep=2000,
        model_path="models/ppo.pth"
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.update_timestep = update_timestep
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build policy-value network and optimizer
        self.policy_value_net = PolicyValueNetwork(
            state_size=self.state_size,
            hidden_size=self.hidden_size,
            action_size=self.action_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_value_net.parameters(), lr=self.learning_rate
        )

        # Storage for trajectories
        self.reset_buffer()

    def reset_buffer(self):
        """Clears all buffers storing trajectories."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.timesteps = 0

    def select_action(self, state):
        """
        Given a state (numpy array), returns:
          - action (tensor, shape [action_size])
          - log_prob (tensor scalar)
          - value (tensor scalar)
        """
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        mean, log_std, value = self.policy_value_net(state_tensor)

        # Replace NaNs just in case
        nan_mask = torch.isnan(mean) | torch.isnan(log_std)
        if nan_mask.any():
            mean = torch.where(nan_mask, torch.zeros_like(mean), mean)
            log_std = torch.where(nan_mask, torch.full_like(log_std, -4.0), log_std)

        std = log_std.exp()
        dist = Normal(mean, std)

        # Sample action with reparameterization trick
        action_raw = dist.rsample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
        action_tanh = torch.tanh(action_raw)

        return action_tanh.squeeze(), log_prob.squeeze(), value.squeeze()

    def compute_gae_and_returns(self, next_value):
        """
        Compute Generalized Advantage Estimate (GAE) and discounted returns.
        next_value: value estimate for the state following the last transition
        Returns:
          - returns: tensor of shape [batch_size]
          - advantages: tensor of shape [batch_size]
        """
        rewards = self.rewards
        dones = self.dones
        values = self.values + [next_value.detach()]
        gae = 0
        returns = []
        advantages = []

        # Traverse trajectory in reverse
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            adv = gae
            ret = adv + values[step]
            advantages.insert(0, adv)
            returns.insert(0, ret)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return returns, advantages

    def update(self):
        """
        Perform PPO update using collected trajectories in the buffer.
        """
        # Convert buffers to tensors
        states = torch.stack(self.states).to(self.device)               # shape: [T, state_size]
        actions = torch.stack(self.actions).to(self.device)             # shape: [T, action_size]
        old_log_probs = torch.stack(self.log_probs).to(self.device)     # shape: [T]
        values = torch.stack(self.values).to(self.device)               # shape: [T]

        # Compute value for next state
        with torch.no_grad():
            _, _, next_value = self.policy_value_net(states[-1].unsqueeze(0))
        returns, advantages = self.compute_gae_and_returns(next_value.squeeze())

        # PPO update for k_epochs
        for _ in range(self.k_epochs):
            # Forward pass to get new log_probs, entropy, and value estimates
            mean, log_std, new_values = self.policy_value_net(states)
            std = log_std.exp()
            dist = Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            # Compute ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic (value) loss (MSE between returns and new_values)
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()

            # Total loss
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Clear buffer after update
        self.reset_buffer()

    def train(
        self,
        env,
        num_episodes=1000,
        update_timestep=2000,
        print_freq=100,
        save_freq=100
    ):
        """
        Train the PPO agent.
        - env: Gym-like environment
        - num_episodes: total episodes to train
        - update_timestep: number of environment steps between PPO updates
        - print_freq: how often (in episodes) to print progress
        - save_freq: how often (in episodes) to save checkpoint
        """
        timestep_counter = 0
        total_reward = 0.0
        max_reward = float('-inf')

        for episode in tqdm(range(1, num_episodes + 1), desc="Training PPO Agent"):
            obs = env.reset()
            state = obs
            episode_reward = 0.0
            done = False

            while not done:
                # Select action (tanhed and scaled), get log_prob and value
                action_tanh, log_prob, value = self.select_action(state)

                # Convert action to numpy and scale to environment's action space
                action_np = action_tanh.detach().cpu().numpy()
                action_np = action_np * [
                    env.config["vehicle"]["acceleration"],
                    env.config["vehicle"]["steering"]
                ]

                next_obs, obs_info, reward, done, _ = env.step(action_np)

                # Store transition in buffer
                self.states.append(torch.FloatTensor(state).to(self.device))
                self.actions.append(action_tanh)
                self.log_probs.append(log_prob)
                self.values.append(value)
                self.rewards.append(reward)
                self.dones.append(float(done))

                state = next_obs
                episode_reward += reward
                total_reward += reward
                timestep_counter += 1

                # If we've collected enough timesteps, update PPO
                if timestep_counter >= update_timestep:
                    self.update()
                    timestep_counter = 0

            # Episode done: track performance
            if episode_reward > max_reward:
                max_reward = episode_reward
                print(f"Max reward: {max_reward:.2f} at episode {episode}")
                if episode > num_episodes // 3:
                    self.save_model(self.model_path)

            if episode % print_freq == 0:
                avg_reward = total_reward / episode
                tqdm.write(f"Episode {episode}/{num_episodes} | "
                           f"Max reward: {max_reward:.2f} | "
                           f"Avg reward: {avg_reward:.2f}")

            if episode % save_freq == 0:
                checkpoint_path = f"{self.model_path[:-4]}_ep{episode}.pth"
                self.save_model(checkpoint_path)

        # Final update if there are leftover transitions
        if timestep_counter > 0:
            self.update()

        print(f"Training completed. Avg reward: {total_reward/num_episodes:.2f}")

    def evaluate(self, env, num_episodes=10, top_k=5, video_dir="videos"):
        """
        Evaluate the PPO agent by running episodes and recording the top_k highest-reward runs.
        """
        os.makedirs(video_dir, exist_ok=True)
        all_episodes = []

        for episode in tqdm(range(num_episodes), desc="Evaluating PPO Agent"):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action_tanh, _, _ = self.select_action(state)
                action_np = action_tanh.detach().cpu().numpy()
                action_np = action_np * [
                    env.config["vehicle"]["acceleration"],
                    env.config["vehicle"]["steering"]
                ]
                next_state, reward, done, _ = env.step(action_np)
                total_reward += reward
                state = next_state

            all_episodes.append((total_reward, episode))

        # Sort and replay top_k episodes
        sorted_eps = sorted(all_episodes, key=lambda x: x[0], reverse=True)
        top_eps = sorted_eps[:top_k]

        for reward, episode in top_eps:
            record_env = Recorder(env, path=video_dir,
                                  videoname=f"ppo_episode_{episode}_reward_{reward:.2f}")
            state = record_env.reset()
            done = False
            while not done:
                action_tanh, _, _ = self.select_action(state)
                action_np = action_tanh.detach().cpu().numpy()
                action_np = action_np * [
                    env.config["vehicle"]["acceleration"],
                    env.config["vehicle"]["steering"]
                ]
                next_state, reward, done, _ = record_env.step(action_np)
                state = next_state
            record_env.close()

    def save_model(self, model_path=None):
        """
        Saves the policy-value network's weights to disk.
        """
        path = model_path if model_path is not None else self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_value_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_path=None):
        """
        Loads weights from disk into the policy-value network.
        """
        path = model_path if model_path is not None else self.model_path
        if os.path.exists(path):
            self.policy_value_net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file {path} does not exist.")
