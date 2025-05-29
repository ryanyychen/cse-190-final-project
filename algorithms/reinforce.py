import os
import torch
from tqdm import tqdm
from gym_recorder import Recorder

class REINFORCEAgent:
    def __init__(self, state_size, hidden_size, action_size, learning_rate=0.01, gamma=0.95, model_path="models/reinforce.pth"):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_path=model_path
        self.policy, self.optimizer = self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def build_model(self):
        # Simple neural network for policy approximation
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.action_size * 2),
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer
    
    def select_action(self, state):
        flattened_state = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        output = self.policy(flattened_state)

        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)

        # Replace NaNs while keeping computation in autograd graph
        nan_mask = torch.isnan(mean) | torch.isnan(log_std)
        mean = torch.where(nan_mask, torch.zeros_like(mean), mean)
        log_std = torch.where(nan_mask, torch.full_like(log_std, -4.0), log_std)  # std = exp(-4) ~ 0.018

        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)

        action = distribution.rsample()  # allows gradients through the sample
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)

        return action.squeeze(), log_prob.squeeze()
    
    def compute_discounted_rewards(self, rewards):
        # Compute discounted rewards using gamma
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
        return torch.FloatTensor(discounted_rewards).to(self.device)
    
    def update_policy(self, rewards, log_probs):
        log_probs = log_probs.to(self.device)
        
        # Compute the discounted rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # Normalize the rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Compute the policy loss
        policy_loss = [-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]
        
        # Update the policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train(self, env, num_episodes=1000, print_freq=100, save_freq=100):
        total_reward = 0
        max_reward = float('-inf')
        for episode in tqdm(range(num_episodes), desc="Training REINFORCE Agent"):
            obs, obs_info = env.reset()
            state = obs
            log_probs = []
            rewards = []
            done = False
            steps = 0

            while not done:
                # Select action, take action, and record
                action, log_prob = self.select_action(state)
                # Scale action to range of environment's action space
                action = action.detach().numpy()
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]

                next_obs, obs_info, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_obs
                steps += 1
            
            ep_reward = sum(rewards)
            if (ep_reward > max_reward):
                max_reward = ep_reward
                self.save_model(self.model_path)
            
            # Update policy after each episode
            self.update_policy(rewards, log_probs)
            if (episode + 1) % print_freq == 0:
                tqdm.write(f"Episode {episode + 1}/{num_episodes} | Max reward: {max_reward:.2f} | Avg reward: {total_reward/(episode):.2f}")

            if (episode + 1) % save_freq == 0:
                self.save_model(f"{self.model_path[:-4]}_ep{episode+1}.pth")
        
        print(f"Training completed. Avg reward: {total_reward/num_episodes:.2f}")
    
    def evaluate(self, env, num_episodes=10, top_k=5, video_dir="videos"):
        os.makedirs(video_dir, exist_ok=True)

        all_episodes = []

        for episode in tqdm(range(num_episodes), desc="Evaluating REINFORCE Agent"):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.select_action(state)
                action = action.detach().numpy()
                # Scale action to range of environment's action space
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            
            all_episodes.append((total_reward, episode))

        # Sort episodes by total reward and select top_k
        sorted_episodes = sorted(all_episodes, key=lambda x: x[0], reverse=True)
        top_episodes = sorted_episodes[:top_k]

        for reward, episode in top_episodes:
            record_env = Recorder(env, path=video_dir, videoname=f"reinforce_episode_{episode}_reward_{reward:.2f}")
            state = record_env.reset()
            done = False
            while not done:
                action, _ = self.select_action(state)
                action = action.detach().numpy()
                # Scale action to range of environment's action space
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]

                next_state, reward, done, _ = record_env.step(action)
                state = next_state
            record_env.close()
    
    def save_model(self, model_path="models/reinforce.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/reinforce.pth"):
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
