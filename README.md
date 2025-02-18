# 🏦 Hierarchical Reinforced Trader (HRT)
*A Bi-Level Approach for Optimizing Stock Selection and Execution*

## 📌 Overview
The **Hierarchical Reinforced Trader (HRT)** is a novel stock trading framework that integrates **Hierarchical Reinforcement Learning (HRL)** to optimize both **stock selection** and **trade execution**. 

HRT addresses key challenges in deep reinforcement learning-based trading, including:
- 📈 **Curse of Dimensionality** – Handling a large number of stocks efficiently.
- 🔄 **Inertia in Trading Actions** – Reducing momentum-based biases in trading decisions.
- 📊 **Insufficient Portfolio Diversification** – Enhancing exposure across multiple sectors.

### ✨ Key Features
✅ **Hierarchical Reinforcement Learning (HRL)** framework  
✅ **Proximal Policy Optimization (PPO)** for strategic stock selection  
✅ **Deep Deterministic Policy Gradient (DDPG)** for trade execution  
✅ **Phased Alternating Training** for joint optimization  
✅ **Sentiment & market data integration** (FinGPT & Yahoo Finance)  
✅ **Scalable to large stock pools (S&P 500 tested)**  

---

## 🏗️ Methodology

### 🔹 1. Hierarchical Reinforcement Learning (HRL)
HRT **separates decision-making into two levels**:
- **High-Level Controller (HLC)** → **PPO-based** agent selects stocks to *buy, sell, or hold*.
- **Low-Level Controller (LLC)** → **DDPG-based** agent optimizes the *trade volume* for execution.

### 🔹 2. Phased Alternating Training
- **Step 1:** Train **HLC** independently to learn **trading directions**.
- **Step 2:** Train **LLC** to optimize **trade execution**.
- **Step 3:** Iteratively refine both using **phased alternating training**.

### 🔹 3. Reinforcement Learning Algorithms Used
| Component | Algorithm | Role |
|-----------|----------|------|
| **HLC** | PPO | Strategic Stock Selection |
| **LLC** | DDPG | Trade Execution Optimization |

---

## 📊 Performance Evaluations

### 📂 1. Dataset
- **Stock Data:** S&P 500 (2015-2022) from **Yahoo Finance** (OHLCV + VWAP)
- **Sentiment Analysis:** **FinGPT** (financial news & social media trends)

### 📈 2. Evaluation Metrics
✅ **Cumulative Return**: Total portfolio return  
✅ **Annualized Return & Volatility**: Measuring growth and risk-adjusted performance  
✅ **Sharpe Ratio**: Risk-adjusted return  
✅ **Maximum Drawdown**: Portfolio risk measurement  

### 🏆 3. Key Results
📌 **HRT vs Baselines (DDPG, PPO, S&P 500, Min-Variance Portfolio)**

| Metric | 2021 (Bull Market) | 2022 (Bear Market) |
|--------|-----------------|-----------------|
| **Sharpe Ratio (HRT)** | **2.74** | **0.41** |
| **Sharpe Ratio (S&P 500)** | 2.27 | -0.83 |
| **Max Drawdown (HRT)** | -7.55% | -5.48% |
| **Max Drawdown (S&P 500)** | -5.21% | -25.43% |

📌 **HRT achieves superior returns and lower risk in both market conditions!** 🚀

---

## ⚙️ Installation & Usage

### 1️⃣ **Installation**
```bash
git clone https://github.com/your-repo/HRT.git
cd HRT
pip install -r requirements.txt


