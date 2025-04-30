import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockSelectionEnv(gym.Env):
    def __init__(self, features_df, forward_returns_df, sentiment_df, llc_feedback_fn=None):
        super(StockSelectionEnv, self).__init__()
        self.features_df = features_df.copy()
        self.forward_returns_df = forward_returns_df.copy()
        self.sentiment_df = sentiment_df.copy()

        self.tickers = sorted(self.features_df.columns)
        self.stock_dim = len(self.tickers)
        self.dates = sorted(self.features_df.index)
        self.current_step = 0
        self.llc_feedback_fn = llc_feedback_fn
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.stock_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3] * self.stock_dim)  # Buy, Sell, Hold

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        features = self.features_df.iloc[self.current_step].values
        sentiments = self.sentiment_df.iloc[self.current_step].values
        return np.concatenate([features, sentiments]).astype(np.float32)

    def step(self, actions):
        date = self.dates[self.current_step]
        forward_returns = self.forward_returns_df.loc[date].values
        alignment_rewards = []

        for idx, action in enumerate(actions):
            price_change = forward_returns[idx]
            if action == 1:  # Buy
                reward = np.sign(price_change)
            elif action == -1:  # Sell
                reward = -np.sign(price_change)
            else:  # Hold
                reward = 0
            alignment_rewards.append(reward)

        alignment_reward = np.sum(alignment_rewards)

        llc_reward = 0
        if self.llc_feedback_fn:
            llc_reward = self.llc_feedback_fn(self.current_step)

        alpha_t = np.exp(-0.001 * self.current_step)
        final_reward = alpha_t * alignment_reward + (1 - alpha_t) * llc_reward
        final_reward = final_reward / 1e6 
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        return self._get_state(), final_reward, done, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step}/{len(self.dates)}")
 
