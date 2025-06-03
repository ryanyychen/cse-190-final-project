import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from gym_recorder import Recorder

class REINFORCEAgent:
    def __init__(self, state_size, hidden_size, action_size, learning_rate=0.0001, gamma=0.99, model_path="models/reinforce.pth"):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy, self.optimizer = self.build_model()
        self.policy.to(self.device)
        self.reward_history = []  # Track rewards for adaptive learning
        self.entropy_coef = 0.05  # Initial entropy coefficient
        self.best_recent_avg = float('-inf')  # Track best recent average
        self.consistency_threshold = 0.25  # Target ratio of recent avg to max reward
        self.min_episodes_for_consistency = 100  # Minimum episodes to consider for consistency
        self.no_improvement_threshold = 150  # Episodes without improvement before increasing exploration
        self.exploration_boost = 1.2  # Factor to increase exploration
        self.max_entropy = 0.2  # Cap on maximum entropy coefficient
        self.min_entropy = 0.01  # Minimum entropy coefficient
        self.recent_window = 100  # Window size for recent average calculation
        self.performance_history = []  # Track recent averages for model saving
        self.policy_buffer = []  # Store recent good policies
        self.buffer_size = 5  # Number of policies to keep in buffer
        self.min_reward_for_buffer = 20.0  # Minimum reward to consider for buffer

    def build_model(self):
        # Deeper network with more capacity and dropout
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, self.action_size * 2),
        )

        # Initialize weights with smaller values
        for layer in model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.zeros_(layer.bias)

        # Optimizer with better learning rate and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return model, optimizer

    def select_action(self, state):
        # Move state to GPU and ensure it's the right shape
        if isinstance(state, dict) and 'state' in state:
            state = state['state']
        
        if len(state.shape) > 1:
            state = state.flatten()
        
        flattened_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        output = self.policy(flattened_state)

        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)

        # Replace NaNs while keeping computation in autograd graph
        nan_mask = torch.isnan(mean) | torch.isnan(log_std)
        mean = torch.where(nan_mask, torch.zeros_like(mean), mean)
        log_std = torch.where(nan_mask, torch.full_like(log_std, -4.0), log_std)

        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)

        # Add noise for exploration - more controlled noise scaling
        noise = torch.randn_like(mean) * (0.1 + self.entropy_coef * 0.1)  # Reduced noise scaling
        action = distribution.rsample() + noise
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)

        # Move action to CPU for environment
        return action.squeeze().cpu(), log_prob.squeeze()

    def compute_discounted_rewards(self, rewards):
        # Compute discounted rewards using gamma
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
        return torch.FloatTensor(discounted_rewards).to(self.device)

    def update_policy(self, rewards, log_probs):
        # Move log_probs to GPU
        log_probs = torch.stack(log_probs).to(self.device)
        
        # Compute the discounted rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # Normalize the rewards with better numerical stability
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(unbiased=False) + 1e-8)
        
        # Compute the policy loss with entropy regularization
        policy_loss = -log_probs * discounted_rewards
        entropy = -self.entropy_coef * torch.mean(log_probs)  # Entropy regularization
        loss = policy_loss.sum() - entropy
        
        # Update the policy
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Adjust entropy coefficient based on performance consistency
        if len(self.reward_history) > self.min_episodes_for_consistency:
            recent_avg = sum(self.reward_history[-self.min_episodes_for_consistency:]) / self.min_episodes_for_consistency
            max_reward = max(self.reward_history[-self.min_episodes_for_consistency:])
            
            # Calculate consistency ratio
            consistency_ratio = recent_avg / max_reward if max_reward > 0 else 0
            
            if consistency_ratio < self.consistency_threshold:
                # Increase exploration if performance is inconsistent
                self.entropy_coef *= 1.1
            else:
                # Decrease exploration if performance is consistent
                self.entropy_coef *= 0.99
            
            # Keep entropy coefficient within reasonable bounds
            self.entropy_coef = max(0.01, min(0.2, self.entropy_coef))

    def train(self, env, num_episodes=1000, print_freq=100, save_freq=100):
        total_reward = 0
        max_reward = float('-inf')
        rewards_history = []  # Track rewards for monitoring
        best_reward = float('-inf')
        episode_rewards = []  # Track all episode rewards
        no_improvement_count = 0  # Track episodes without improvement
        last_improvement_episode = 0  # Track when we last saw improvement
        consecutive_resets = 0  # Track number of consecutive resets
        best_recent_avg = float('-inf')  # Track best recent average for model saving

        for episode in tqdm(range(num_episodes), desc="Training REINFORCE Agent"):
            obs, obs_info = env.reset()
            state = obs
            log_probs = []
            rewards = []
            done = False
            steps = 0

            while not done:
                action, log_prob = self.select_action(state)
                action = action.detach().numpy()
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]

                next_obs, obs_info, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_obs
                steps += 1
            
            ep_reward = sum(rewards)
            rewards_history.append(ep_reward)
            episode_rewards.append(ep_reward)
            
            # Calculate recent average and consistency metrics
            if len(rewards_history) >= self.recent_window:
                recent_avg = sum(rewards_history[-self.recent_window:]) / self.recent_window
                max_recent = max(rewards_history[-self.recent_window:])
                consistency_ratio = recent_avg / max_recent if max_recent > 0 else 0
                
                # Save model if we have a new best recent average
                if recent_avg > best_recent_avg:
                    best_recent_avg = recent_avg
                    print(f"New best recent average: {recent_avg:.2f} (consistency ratio: {consistency_ratio:.2f})")
                    self.save_model(f"{self.model_path[:-4]}_best.pth")
                    last_improvement_episode = episode
                    consecutive_resets = 0  # Reset counter on improvement
                    no_improvement_count = 0  # Reset no improvement counter
            
            if (ep_reward > max_reward):
                max_reward = ep_reward
                print(f"Max reward: {max_reward:.2f} at episode {episode + 1}")
            
            # Update policy after each episode
            self.update_policy(rewards, log_probs)
            
            # More balanced exploration strategy
            if no_improvement_count > self.no_improvement_threshold:
                self.entropy_coef = min(self.max_entropy, self.entropy_coef * self.exploration_boost)
                no_improvement_count = 0
                print(f"Increasing exploration (entropy coef: {self.entropy_coef:.3f})")
            
            # Reset entropy coefficient if we've been stuck for too long
            if episode - last_improvement_episode > 500:
                consecutive_resets += 1
                if consecutive_resets < 3:  # Limit number of resets
                    self.entropy_coef = max(self.min_entropy, self.entropy_coef * 0.5)  # Gradual reduction
                    print(f"Reducing exploration due to stagnation (entropy coef: {self.entropy_coef:.3f})")
                    last_improvement_episode = episode
            
            if (episode + 1) % print_freq == 0:
                avg_reward = total_reward / (episode + 1)
                recent_avg = sum(rewards_history[-print_freq:]) / print_freq
                max_recent = max(rewards_history[-print_freq:])
                consistency_ratio = recent_avg / max_recent if max_recent > 0 else 0
                tqdm.write(f"Episode {episode + 1}/{num_episodes} | Max reward: {max_reward:.2f} | "
                          f"Avg reward: {avg_reward:.2f} | Recent avg: {recent_avg:.2f} | "
                          f"Consistency ratio: {consistency_ratio:.2f} | Entropy coef: {self.entropy_coef:.3f}")

            if (episode + 1) % save_freq == 0:
                self.save_model(f"{self.model_path[:-4]}_ep{episode+1}.pth")
        
        print(f"Training completed. Avg reward: {total_reward/num_episodes:.2f}")
        return rewards_history
    
    def evaluate(self, env, num_episodes=10, top_k=5, video_dir="videos"):
        os.makedirs(video_dir, exist_ok=True)
        all_episodes = []
        episode_frames = {}  # Store frames for each episode

        try:
            # First run to collect rewards and frames
            for episode in tqdm(range(num_episodes), desc="Evaluating REINFORCE Agent"):
                state = env.reset()
                done = False
                total_reward = 0
                frames = []
                step_count = 0

                try:
                    while not done:
                        # Get frame before action
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                        
                        # Take action
                        action, _ = self.select_action(state)
                        action = action.detach().numpy()
                        # Scale action to range of environment's action space
                        action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]
                        
                        next_state, reward, done, _ = env.step(action)
                        total_reward += reward
                        state = next_state
                        step_count += 1

                        # Force minimum episode length
                        if step_count < 30:  # Minimum 30 steps per episode
                            done = False

                    all_episodes.append((total_reward, episode))
                    episode_frames[episode] = frames
                    print(f"Episode {episode} completed with reward {total_reward:.2f} and {len(frames)} frames")

                except KeyboardInterrupt:
                    print("\nEvaluation interrupted by user. Saving current progress...")
                    break
                except Exception as e:
                    print(f"Error during episode {episode}: {str(e)}")
                    continue

            # Sort episodes by total reward and select top_k
            if all_episodes:  # Only proceed if we have episodes to process
                sorted_episodes = sorted(all_episodes, key=lambda x: x[0], reverse=True)
                top_episodes = sorted_episodes[:min(top_k, len(sorted_episodes))]

                # Save videos for top episodes
                for reward, episode in top_episodes:
                    try:
                        frames = episode_frames.get(episode, [])
                        if not frames:
                            print(f"No frames available for episode {episode}")
                            continue

                        height, width, layers = frames[0].shape
                        video_path = os.path.join(video_dir, f"reinforce_episode_{episode}_reward_{reward:.2f}.mp4")
                        
                        # Use a lower frame rate for smoother playback
                        fps = 5  # Reduced from 10 to 5 for smoother playback
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        
                        # Write frames to video
                        for frame in frames:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        
                        out.release()
                        print(f"Saved video to {video_path} with {len(frames)} frames at {fps} fps")

                    except Exception as e:
                        print(f"Error saving video for episode {episode}: {str(e)}")
                        continue

            print("Evaluation completed successfully")
            return all_episodes  # Return the episodes data for analysis

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user. Cleaning up...")
            # Clean up any open video writers
            for episode in episode_frames:
                if 'out' in locals():
                    out.release()
            return all_episodes  # Return whatever episodes we managed to complete
        except Exception as e:
            print(f"Unexpected error during evaluation: {str(e)}")
            return all_episodes  # Return whatever episodes we managed to complete
    
    def save_model(self, model_path="models/reinforce.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'reward_history': self.reward_history
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/reinforce.pth"):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.entropy_coef = checkpoint.get('entropy_coef', 0.05)
            self.reward_history = checkpoint.get('reward_history', [])
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
