import pandas as pd
import numpy as np
from torch.optim import AdamW
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from envs.stock_trading_env import StockSelectionEnv
from envs.stock_execution_env import StockExecutionEnv

# Custom callback to track rewards
class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_rewards = []
        
    def _on_step(self) -> bool:
        if "episode" in self.locals:
            for info in self.locals["episode"]:
                if "r" in info:
                    self.current_episode_rewards.append(info["r"])
        
        if self.locals.get("dones", False):
            if self.current_episode_rewards:
                mean_reward = np.mean(self.current_episode_rewards)
                self.episode_rewards.append(mean_reward)
                self.logger.record("custom/mean_ep_reward", mean_reward)
                self.current_episode_rewards = []
        
        return True

def main():
    # ✅ Load data
    features_df = pd.read_csv('data/features.csv', index_col=0)
    forward_returns_df = pd.read_csv('data/forward_returns.csv', index_col=0)
    sentiment_df = pd.read_csv('data/sentiment_dummy.csv', index_col=0)
    stock_raw = pd.read_csv('data/stock_data.csv')
    price_df = stock_raw.pivot(index="Date", columns="Ticker", values="Close")

    # ✅ Load trained LLC model
    llc_model = DDPG.load('models/llc_ddpg')

    # ✅ Initialize execution environment for LLC to compute rewards
    llc_env = StockExecutionEnv(price_df)
    
    # ✅ Real LLC feedback function
    def real_llc_feedback(step):
        # Reset the LLC env to the current step
        llc_env.current_step = step
        
        # Sample a dummy HLC decision: for now assume all 0s (hold) (you can pass real HLC actions here)
        hlc_decisions = np.zeros(llc_env.stock_dim)
        llc_env.set_hlc_decisions(hlc_decisions)

        state = llc_env._get_state()
        action, _ = llc_model.predict(state, deterministic=True)
        
        # Simulate one step in LLC env
        next_state, reward, done, _ = llc_env.step(action)

        return reward

    # ✅ Initialize HLC environment
    env = StockSelectionEnv(
        features_df, 
        forward_returns_df, 
        sentiment_df, 
        llc_feedback_fn=real_llc_feedback
    )

    # ✅ Initialize callback for tracking rewards
    reward_callback = RewardTrackerCallback()

    # ✅ Initialize PPO model

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        clip_range=0.2,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./logs/hlc_tensorboard/",
        policy_kwargs={"optimizer_class": AdamW}
    )


    # 🚀 Train model with proper callback
    total_steps = 100_000
    model.learn(
        total_timesteps=total_steps,
        callback=reward_callback,
        progress_bar=True
    )

    # ✅ Save the model
    model.save('models/hlc_ppo')
    print("✅ HLC PPO model saved!")

    # ✅ Plot rewards
    plt.figure(figsize=(12, 6))
    
    if len(reward_callback.episode_rewards) > 0:
        plt.plot(reward_callback.episode_rewards, alpha=0.3, label='Raw Rewards', color='blue')

        window_size = max(1, len(reward_callback.episode_rewards) // 20)
        smoothed_rewards = pd.Series(reward_callback.episode_rewards).rolling(
            window=window_size, 
            min_periods=1
        ).mean()
        
        plt.plot(smoothed_rewards, label=f'Smoothed (window={window_size})', color='red', linewidth=2)
        
        plt.title('HLC PPO Training Rewards (with LLC feedback)')
        plt.xlabel('Episode')
        plt.ylabel('Mean Episode Reward')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('results/hlc_training_rewards_with_llc.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️ No rewards were recorded during training")

if __name__ == "__main__":
    main()
