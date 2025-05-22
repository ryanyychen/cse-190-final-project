import torch

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
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
        # Greedily select action based on the policy
        print(state)
        probs = self.policy(state)
        action = torch.argmax(probs, dim=1)
        return action.item()
    
    def update_policy(self, rewards, log_probs):
        # Compute the discounted rewards
        discounted_rewards = []
        for t in range(len(rewards)):
            G = sum([rewards[i] * (0.99 ** (i - t)) for i in range(t, len(rewards))])
            discounted_rewards.append(G)
        
        # Normalize the rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)
        
        # Compute the policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        # Update the policy
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                # e-greedy action selection
                if torch.rand(1).item() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store log probability and reward
                log_prob = torch.log(self.policy(torch.FloatTensor(state).unsqueeze(0))[0][action])
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
            
            # Update policy after each episode
            self.update_policy(rewards, log_probs)
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes} completed.")