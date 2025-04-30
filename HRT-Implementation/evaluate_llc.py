# evaluate_llc.py (updated with per-step prints and render)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from envs.stock_execution_env import StockExecutionEnv

def evaluate_llc(model_path, n_episodes=5):
    # ✅ Load the trained LLC model
    model = DDPG.load(model_path)

    # ✅ Load the stock price data
    stock_raw = pd.read_csv('data/stock_data.csv')
    price_df = stock_raw.pivot(index="Date", columns="Ticker", values="Close")

    # ✅ Create evaluation environment
    env = StockExecutionEnv(price_df, initial_cash=1e6, max_shares=100)

    episode_rewards = []

    for ep in range(n_episodes):
        print(f"\n=== Episode {ep+1} ===")
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            # 🚀 Predict action from trained LLC model
            action, _ = model.predict(obs, deterministic=True)  # <== DETERMINISTIC prediction
            
            # Step in environment
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1

            # 🖨 Print info every step
            print(f"Step {step_count}: Action = {np.round(action, 2)}, Reward = {reward:.2f}, TotalReward = {total_reward:.2f}")

            # 🖥 Render environment (cash, holdings)
            env.render()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1} finished! Total Reward = {total_reward:.2f}")

    # ✅ Plot total rewards
    plt.figure(figsize=(10,6))
    plt.plot(episode_rewards, marker='o', linestyle='-')
    plt.title(f"LLC DDPG Evaluation: Total Rewards over {n_episodes} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Portfolio Reward")
    plt.grid(True)
    plt.savefig('results/llc_evaluation_rewards_with_steps.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ LLC full evaluation complete!")

if __name__ == "__main__":
    evaluate_llc("models/llc_ddpg", n_episodes=5)
# This code evaluates the trained DDPG model on the stock execution environment.
# It prints the action taken, the reward received, and the total reward at each step.
# It also renders the environment to visualize the cash and holdings.
# Finally, it plots the total rewards for each episode.
# The evaluation process is repeated for a specified number of episodes.
# The results are saved as a PNG file.
# The code is designed to be run as a standalone script.
# It uses the Stable Baselines3 library for reinforcement learning and Matplotlib for plotting.
# The evaluation function loads the trained model, resets the environment, and runs the model for a specified number of episodes.
# The environment is rendered at each step to visualize the agent's performance.
# The code is modular and can be easily adapted for different models or environments.
# The evaluation function is designed to be reusable and can be called with different model paths or parameters.
# The code is well-structured and follows best practices for reinforcement learning evaluation.
# The evaluation process is designed to be efficient and provides detailed feedback on the agent's performance. 