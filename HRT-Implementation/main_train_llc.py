import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added this import
from collections import deque
import random
from envs.stock_execution_env import StockExecutionEnv
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))  # Output between [-1, 1]
        return x * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = F.relu(self.layer1(xu))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.tau = 0.005
        self.gamma = 0.99
        self.max_action = max_action
        self.noise_std = 0.1

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise:
            action = (action + np.random.normal(0, self.noise_std, size=action.shape))
        return action.clip(-self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
        action = torch.FloatTensor(np.array([t[1] for t in batch])).to(device)
        reward = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
        done = torch.FloatTensor(np.array([1 if t[4] else 0 for t in batch])).unsqueeze(1).to(device)

        # Compute target Q
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.gamma * target_Q.detach()

        # Compute current Q
        current_Q = self.critic(state, action)

        # Critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))

def train_llc():
    # Load data
    stock_raw = pd.read_csv('/home/govind/rl/HRT_Project/Different-approach/data/Processed_training_intraday_data.csv')
    price_df = stock_raw.pivot(index="Date", columns="Ticker", values="Close")
    
    # Initialize environment
    env = StockExecutionEnv(price_df, initial_cash=1e6, max_shares=100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize DDPG agent
    agent = DDPG(state_dim, action_dim, max_action)
    
    # Training parameters
    episodes = 1000
    max_steps = len(env.dates) - 1
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        
        # Generate random HLC decisions for this episode
        hlc_decisions = np.random.choice([-1, 0, 1], size=env.stock_dim)
        env.set_hlc_decisions(hlc_decisions)
        
        episode_reward = 0
        episode_value = env.initial_cash
        
        for step in range(max_steps):
            # Select action with exploration noise
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # Train agent
            agent.train()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Print progress
        current_value = env.cash + np.sum(env.holdings * env.price_df.iloc[env.current_step].values)
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:,.2f} | Portfolio Value: ${current_value:,.2f}")
    
    # Save trained models
    agent.save("models/llc_ddpg")
    print("✅ LLC training completed and models saved")

if __name__ == "__main__":
    train_llc()
