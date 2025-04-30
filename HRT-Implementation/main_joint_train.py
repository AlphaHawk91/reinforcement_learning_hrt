import os
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from envs.stock_trading_env import StockSelectionEnv
from envs.stock_execution_env import StockExecutionEnv
import matplotlib.pyplot as plt

# Phase settings
HLC_TIMESTEPS = 50_000
LLC_TIMESTEPS = 50_000
REFINEMENT_PHASES = 5
PHASE_TIMESTEPS = 20_000

# Load datasets
features_df = pd.read_csv("data/features.csv", index_col=0)
forward_returns_df = pd.read_csv("data/forward_returns.csv", index_col=0)
sentiment_df = pd.read_csv("data/sentiment_dummy.csv", index_col=0)
stock_df = pd.read_csv("data/stock_data.csv")
price_df = stock_df.pivot(index="Date", columns="Ticker", values="Close")

# ---------- Phase 1: Train HLC (PPO) only with alignment rewards ----------
def train_hlc_model():
    print("\n🚀 Training High-Level Controller (HLC)...")

    def dummy_llc_feedback(step):
        return 0.0  # No LLC feedback during initial HLC phase

    env = DummyVecEnv([lambda: StockSelectionEnv(features_df, forward_returns_df, sentiment_df, dummy_llc_feedback)])
    model = PPO("MlpPolicy", env, verbose=1, batch_size=64)
    model.learn(total_timesteps=HLC_TIMESTEPS)
    model.save("models/hlc_ppo_phase1")
    print("✅ HLC trained and saved!")
    return model

# ---------- Phase 2: Train LLC (DDPG) with frozen HLC ----------
def train_llc_model():
    print("\n🚀 Training Low-Level Controller (LLC)...")
    env = DummyVecEnv([lambda: StockExecutionEnv(price_df, initial_cash=1e6, max_shares=100)])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        action_noise=action_noise,
        buffer_size=200_000,
        learning_rate=1e-3,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000
    )
    model.learn(total_timesteps=LLC_TIMESTEPS)
    model.save("models/llc_ddpg_phase1")
    print("✅ LLC trained and saved!")
    return model

# ---------- Phase 3: Alternating Refinement ----------
def alternating_refinement(hlc_model, llc_model):
    print("\n🔁 Alternating training HLC and LLC with reward blending...")

    hlc_rewards = []
    llc_rewards = []

    for phase in range(REFINEMENT_PHASES):
        print(f"\n🔥 Refinement Round {phase+1}/{REFINEMENT_PHASES}")

        # Blending factor α_t decays over time
        alpha_t = np.exp(-0.001 * phase)

        # ---- HLC Training with LLC reward influence ----
        def blended_llc_feedback(step):
            # Dummy feedback placeholder
            return 100.0 * (1 - alpha_t)

        hlc_env = DummyVecEnv([lambda: StockSelectionEnv(features_df, forward_returns_df, sentiment_df, blended_llc_feedback)])
        hlc_model.set_env(hlc_env)
        hlc_model.learn(total_timesteps=PHASE_TIMESTEPS, reset_num_timesteps=False)
        hlc_model.save("models/hlc_ppo_latest")

        # ---- LLC Training ----
        llc_env = DummyVecEnv([lambda: StockExecutionEnv(price_df, initial_cash=1e6, max_shares=100)])
        llc_model.set_env(llc_env)
        llc_model.learn(total_timesteps=PHASE_TIMESTEPS, reset_num_timesteps=False)
        llc_model.save("models/llc_ddpg_latest")

        hlc_rewards.append(alpha_t)
        llc_rewards.append(1 - alpha_t)

    print("\n✅ Alternating refinement complete!")
    return hlc_rewards, llc_rewards

# ---------- Run full training pipeline ----------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    hlc_model = train_hlc_model()
    llc_model = train_llc_model()
    hlc_rewards, llc_rewards = alternating_refinement(hlc_model, llc_model)

    # ✅ Plot alpha_t decay over time (blending factor)
    plt.plot(hlc_rewards, label="HLC weight (alpha_t)", color="blue")
    plt.plot(llc_rewards, label="LLC weight (1 - alpha_t)", color="red")
    plt.title("Reward Blending over Alternating Phases")
    plt.xlabel("Refinement Phase")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/reward_blending.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
