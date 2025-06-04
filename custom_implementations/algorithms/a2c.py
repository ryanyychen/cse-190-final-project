import os
import torch
from tqdm import tqdm
from gym_recorder import Recorder

class ActorCritic(torch.nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.shared = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU()
                )
                self.actor = torch.nn.Linear(hidden_size, output_size * 2)
                self.critic = torch.nn.Linear(hidden_size, 1)

            def forward(self, x):
                x = self.shared(x)
                actor_out = self.actor(x)
                value = self.critic(x)
                return actor_out, value

class A2CAgent:
    def __init__(self, state_size, hidden_size, action_size, learning_rate=0.01, gamma=0.95, model_path="models/a2c.pth"):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_path = model_path
        self.model, self.optimizer = self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def build_model(self):
        model = ActorCritic(self.state_size, self.hidden_size, self.action_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def select_action(self, state):
        state = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        output, value = self.model(state)

        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)

        nan_mask = torch.isnan(mean) | torch.isnan(log_std)
        mean = torch.where(nan_mask, torch.zeros_like(mean), mean)
        log_std = torch.where(nan_mask, torch.full_like(log_std, -4.0), log_std)

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)

        return action.squeeze(), log_prob.squeeze(), value.squeeze()

    def compute_returns(self, rewards, last_value, dones):
        returns = []
        R = last_value
        # Q: why r, done below?
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        # Q: float tensor?
        return torch.tensor(returns).to(self.device)

    def update(self, log_probs, values, rewards, dones):
        # Q: no_grad may not be necessary
        with torch.no_grad():
            #Q: don't understand what this line is doing
            next_value = values[-1] if len(values) > len(rewards) else torch.tensor(0.0).to(self.device)
            returns = self.compute_returns(rewards, next_value, dones)
        values = torch.stack(values[:-1]) if len(values) > len(rewards) else torch.stack(values)
        log_probs = torch.stack(log_probs)
        
        advantages = returns - values
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        
        #Q: Check the ordering of zero_grad() 
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train(self, env, num_episodes=1000, print_freq=100, save_freq=100):
        total_reward = 0
        max_reward = float('-inf')
        for episode in tqdm(range(num_episodes), desc="Training A2C Agent"):
            obs, _ = env.reset()
            state = obs
            log_probs, rewards, values, dones = [], [], [], []
            done = False

            while not done:
                action, log_prob, value = self.select_action(state)
                values.append(value)
                # action_np = action.detach().cpu().numpy()
                # Q: I didn't move this action to CPU
                action_np = action.detach().numpy()
                action_scaled = action_np * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]

                next_obs, _, reward, done, _ = env.step(action_scaled)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)

                state = next_obs
                total_reward += reward

            # Q: What is this values business
            values.append(torch.tensor(0.0).to(self.device))  # last value = 0 if episode ended
            ep_reward = sum(rewards)
            if ep_reward > max_reward:
                max_reward = ep_reward
                print(f"Max reward: {max_reward:.2f} at episode {episode + 1}")
                if (episode + 1) > num_episodes // 3:
                    self.save_model()
            
            # Update policy after each episode
            self.update(log_probs, values, rewards, dones)
            if (episode + 1) % print_freq == 0:
                tqdm.write(f"Episode {episode + 1}/{num_episodes} | Max reward: {max_reward:.2f} | Avg reward: {total_reward / (episode + 1):.2f}")
            if (episode + 1) % save_freq == 0:
                self.save_model(f"{self.model_path[:-4]}_ep{episode + 1}.pth")

        print(f"Training completed. Avg reward: {total_reward / num_episodes:.2f}")

    def evaluate(self, env, num_episodes=10, top_k=5, video_dir="videos"):
        os.makedirs(video_dir, exist_ok=True)
        all_episodes = []

        for episode in tqdm(range(num_episodes), desc="Evaluating A2C Agent"):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _, _ = self.select_action(state)
                # action = action.detach().cpu().numpy()
                # Q: removed .cpu() call
                action = action.detach().numpy()
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

            all_episodes.append((total_reward, episode))

        sorted_episodes = sorted(all_episodes, key=lambda x: x[0], reverse=True)
        top_episodes = sorted_episodes[:top_k]

        for reward, episode in top_episodes:
            record_env = Recorder(env, path=video_dir, videoname=f"a2c_episode_{episode}_reward_{reward:.2f}")
            state = record_env.reset()
            done = False
            while not done:
                action, _, _ = self.select_action(state)
                # Q: removed .cpu() call
                # action = action.detach().cpu().numpy()
                action = action.detach().numpy()
                action = action * [env.config["vehicle"]["acceleration"], env.config["vehicle"]["steering"]]
                next_state, reward, done, _ = record_env.step(action)
                state = next_state
            record_env.close()

    def save_model(self, model_path=None):
        path = model_path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_path=None):
        # Q: below line necessary?
        path = model_path or self.model_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file {path} does not exist.")
