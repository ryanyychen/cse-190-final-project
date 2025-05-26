import os
import torch
from tqdm import tqdm
from gym_recorder import Recorder

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy, self.optimizer = self.build_model()

    def build_model(self):
        # Simple neural network for policy approximation
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, self.action_size),
            torch.nn.Softmax(dim=-1)
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer
    
    def select_action(self, state):
        # Select action based on distribution
        flattened_state = torch.FloatTensor(state).flatten().unsqueeze(0)
        probs = self.policy(flattened_state)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
        return torch.FloatTensor(discounted_rewards)
    
    def update_policy(self, rewards, log_probs):
        # Compute the discounted rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # Normalize the rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Compute the policy loss
        policy_loss = [-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]
        
        # Update the policy
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=1000):
        total_reward = 0
        for episode in tqdm(range(num_episodes), desc="Training REINFORCE Agent"):
            obs, obs_info = env.reset()
            state = obs
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                # Select action, take action, and record
                action, log_prob = self.select_action(state)
                next_obs, obs_info, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_obs
            
            # Update policy after each episode
            self.update_policy(rewards, log_probs)
            if (episode + 1) % 100 == 0:
                tqdm.write(f"Episode {episode + 1}/{num_episodes} | Avg reward: {total_reward/(episode):.2f}")
        
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
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            
            all_episodes.append((total_reward, episode))

        sorted_episodes = sorted(all_episodes, key=lambda x: x[0], reverse=True)
        top_episodes = sorted_episodes[:top_k]

        for reward, episode in top_episodes:
            record_env = Recorder(env, path=video_dir, videoname=f"reinforce_episode_{episode}_reward_{reward:.2f}")
            state = record_env.reset()
            done = False
            while not done:
                action, _ = self.select_action(state)
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
