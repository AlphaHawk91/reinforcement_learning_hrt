import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockExecutionEnv(gym.Env):
    def __init__(self, price_df, initial_cash=1e6, max_shares=100):
        super().__init__()
        
        # Data preparation
        self.price_df = price_df.copy()
        self.dates = sorted(price_df.index.unique())
        self.tickers = list(price_df.columns)
        self.stock_dim = len(self.tickers)
        
        # Normalize prices (per-stock normalization)
        self.scaler = MinMaxScaler()
        self.normalized_prices = self.scaler.fit_transform(self.price_df)
        
        # Trading parameters
        self.initial_cash = initial_cash
        self.max_shares = max_shares
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.stock_dim)
        self.hlc_decisions = np.zeros(self.stock_dim)  # -1 (sell), 0 (hold), 1 (buy)

        # Action space: continuous [-1, 1] for each stock (will be scaled to [0, max_shares])
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)

        # Observation space: normalized prices + normalized holdings + cash + HLC decisions
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(3 * self.stock_dim + 1,),  # norm_prices + norm_holdings + hlc_decisions + norm_cash
            dtype=np.float32
        )

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.stock_dim)
        self.current_step = 0
        self.hlc_decisions = np.zeros(self.stock_dim)
        return self._get_state()

    def set_hlc_decisions(self, hlc_decisions):
        """Set HLC decisions (-1, 0, 1 for sell/hold/buy)"""
        self.hlc_decisions = np.array(hlc_decisions)

    def _normalize_cash(self, cash):
        """Normalize cash to [-1, 1] range"""
        return (cash - self.initial_cash/2) / (self.initial_cash/2)

    def _normalize_holdings(self, holdings):
        """Normalize holdings to [-1, 1] range"""
        return holdings / self.max_shares

    def _get_state(self):
        # Get current normalized prices
        norm_prices = self.normalized_prices[self.current_step]
        
        # Normalize other components
        norm_holdings = self._normalize_holdings(self.holdings)
        norm_cash = self._normalize_cash(self.cash)
        
        # HLC decisions are already in [-1, 0, 1]
        state = np.concatenate([
            norm_prices,
            norm_holdings,
            self.hlc_decisions,
            [norm_cash]
        ])
        return state.astype(np.float32)

    def step(self, actions):
        # Clip actions to [-1, 1] range
        actions = np.clip(actions, -1, 1)
        current_prices = self.price_df.iloc[self.current_step].values
        
        # Convert actions to actual share amounts (positive only)
        share_amounts = np.round((actions + 1) / 2 * self.max_shares)  # Scale [0, max_shares]
        
        # Execute trades based on HLC decisions
        for idx, (decision, shares) in enumerate(zip(self.hlc_decisions, share_amounts)):
            if decision == 1:  # Buy
                cost = shares * current_prices[idx]
                if self.cash >= cost and shares > 0:
                    self.cash -= cost
                    self.holdings[idx] += shares
            elif decision == -1:  # Sell
                shares = min(shares, self.holdings[idx])
                if shares > 0:
                    self.cash += shares * current_prices[idx]
                    self.holdings[idx] -= shares
            # No action for hold (decision == 0)

        # Move to next time step
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        # Calculate reward as change in portfolio value
        next_prices = self.price_df.iloc[self.current_step].values
        current_value = self.cash + np.sum(self.holdings * current_prices)
        next_value = self.cash + np.sum(self.holdings * next_prices)

        # if current_value < 1e-6:  # Essentially zero
        #     reward = 0.0
        # else:
        #     reward = (next_value - current_value) / current_value  # Percentage return
        reward = (next_value - current_value) / current_value  # Percentage return

        # reward = next_value - current_value



        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        current_prices = self.price_df.iloc[self.current_step].values
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        print(f"\nStep: {self.current_step}/{len(self.dates)-1}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print("Holdings:")
        for ticker, shares, price in zip(self.tickers, self.holdings, current_prices):
            if shares > 0:
                print(f"  {ticker}: {shares} shares @ ${price:.2f} (${shares*price:,.2f})")
        
        print("\nHLC Decisions:")
        print("  Buy:", [self.tickers[i] for i, d in enumerate(self.hlc_decisions) if d == 1])
        print("  Sell:", [self.tickers[i] for i, d in enumerate(self.hlc_decisions) if d == -1])
        print("  Hold:", [self.tickers[i] for i, d in enumerate(self.hlc_decisions) if d == 0])
