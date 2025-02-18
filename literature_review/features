# 📌 Features in Reinforcement Learning for Quantitative Finance
*(Extracted from Section 4.2 of the Survey Paper)*

## 🔍 Overview
Financial market modeling presents challenges due to its **randomness and complexity**. The effectiveness of **Reinforcement Learning (RL) models** depends significantly on the **choice and extraction of input data and features**. This section highlights the key features used in RL-based financial applications.

---

## 📊 Key Feature Categories

### 1️⃣ Price History
- **Common Features**:
  - Open-High-Low-Close (OHLC) data.
  - OHLCV (OHLC + Volume).
  - 20-day lagged returns.
  - Co-integration-based features.
- **Challenges**:
  - **Redundancy**: High correlation among OHLC features can introduce noise.
  - **Underperformance**: Using only a single price feature can be less effective than benchmark strategies.
- **Recent Advancements**:
  - Incorporation of **volatility measures** for **regime-switching detection**.
  - Use of **GARCH models** to capture price fluctuations.

---

### 2️⃣ Technical Analysis
- **Common Indicators**:
  - **Moving Average (MA), Exponential Moving Average (EMA)**.
  - **MACD (Moving Average Convergence Divergence)**.
  - **Japanese Candlestick Patterns**.
  - **Relative Strength Index (RSI)**.
- **Debates**:
  - Some studies challenge the **Efficient Market Hypothesis (EMH)**, arguing that **past prices can predict future trends**.
  - However, the effectiveness of technical indicators in **RL applications is still debated**.

---

### 3️⃣ Fundamental Data & Factor Investing
- **Factors Used**:
  - **Price Momentum** (e.g., stocks that performed well continue to do so).
  - **Value Investing** (e.g., buying stocks that are undervalued).
  - **Size** (e.g., small-cap stocks tend to outperform).
  - **Quality** (e.g., based on financial statements).
  - **Low Beta** (e.g., stocks with low volatility perform better).
- **Challenges**:
  - **Quarterly updates** of accounting data **limit its effectiveness for daily trading signals**.
  - **Limited adoption** in RL models due to **delayed data availability**.

---

### 4️⃣ Order Book Data (LOB)
- **Common Features**:
  - **Order imbalance-based features**.
  - **Depth of market**.
  - **Bid-Ask spread**.
  - **Time to fill a limit order**.
- **Use Cases**:
  - **Trade execution** and **market making**.
  - LOB data provides **fine-grained price movement insights**, making it **valuable for high-frequency trading (HFT)**.

---

### 5️⃣ Sentiment Data
- **Common Sources**:
  - **Reuters News Corpus**.
  - **Twitter & Social Media**.
  - **Thomson Reuters News Analytics**.
- **Benefits**:
  - Reduces **uncertainty in financial models**.
  - Helps **forecast stock price movements**.
- **Challenges**:
  - Sentiment data is **mainly focused on well-known companies**.
  - **Underutilized** in RL-based financial models.

---

### 6️⃣ Macroeconomic Data
- **Common Indicators**:
  - **Interest rates**.
  - **Inflation rates**.
  - **GDP growth**.
  - **Treasury yield curve slopes**.
- **Use Cases**:
  - Useful for modeling **economic cycles and stock trends**.
  - Rarely used in RL models, but some studies incorporate it as a **risk indicator**.

---

### 7️⃣ Other Features
- **Trade Execution Features**:
  - **Elapsed time, remaining shares to order, current time**.
- **Advanced Feature Engineering**:
  - **Volume-weighted average price (VWAP)**.
  - **Credit spreads of corporate bonds**.
- **Emerging Data Sources**:
  - **ESG (Environmental, Social, and Governance) data**.
  - **Supply chain data** for deeper market insights.

---

## 🚀 Summary
1. **Price-based features are widely used but must be combined with other indicators** to improve effectiveness.
2. **Technical indicators and fundamental data provide additional insights but have mixed success in RL models**.
3. **Order book and sentiment data enhance predictive power but are underutilized**.
4. **Macroeconomic factors and alternative data sources are emerging areas for RL in finance**.

---

### 📌 References
- Extracted from **Section 4.2 (Features) of the Review Paper**.
