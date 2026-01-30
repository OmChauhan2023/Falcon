# Algorithmic Trading System: Technical Documentation
**Strategy Name:** FINNIFTY-SMC-XGBoost
**Asset Class:** FINNIFTY

---

## 1. Executive Summary
This document details the architecture, logic, and performance of a hybrid algorithmic trading system designed for **FINNIFTY**. 

The system leverages a "Grey Box" approach, combining deterministic technical structure analysis (Smart Money Concepts) with probabilistic Machine Learning models (XGBoost Classifier) to filter and execute high-probability trade setups.

### Key Objectives:
* **Identify** structural liquidity points (FVGs, Order Blocks).
* **Validate** entries using Time-Series Machine Learning.
* **Manage** risk dynamically using Chandelier Exits and ATR.

---

## 2. Theoretical Framework & Strategy Logic

### 2.1 Core Market Structure (The "Alpha")
The strategy is grounded in **Smart Money Concepts (SMC)**, specifically targeting:

1.  **Optimal Trade Entry (OTE):**
    * Retracement levels measured via Fibonacci sequences.
    * **Key Zone:** 0.50 to 0.618 retracement.
    
2.  **Fair Value Gaps (FVG):**
    * Identifies inefficiencies in price delivery where buying/selling pressure was imbalanced.
    * Acts as a magnet for price rebalancing.

3.  **Market Regime Classification:**
    * Separates market states into *Trending* vs. *Ranging* to adjust signal sensitivity.

### 2.2 Machine Learning Overlay
To reduce false positives common in pure technical strategies, a supervised learning layer is applied.

* **Algorithm:** XGBoost Classifier 
* **Target:** Binary Classification (1 = Profitable Up Move, 0 = Down/Neutral)
* **Input Features:** 5 distinct features including price structure, volatility, and momentum.

---

## 3. Data Pipeline & Feature Engineering

### 3.1 Data Architecture
The system utilizes a split-validation approach to prevent look-ahead bias:

| Set | Purpose | File Count | Timeline |
| :--- | :--- | :--- | :--- |
| **Training** | Model Fitting | 12 | 2021-2023 |
| **Validation** | Tuning & Thresholds | 2 | 2023 Early |
| **Testing** | Out-of-Sample Performance | 3 | 2023-2024 |

### 3.2 Key Feature Importance
The ML model identified the following as the most critical predictors of price movement:

1. **bullish_fvg_proximity**
2. **bearish_fvg_proximity**
3. **regime_strong_down**
4. **ema_50_slope**
5. **is_opening_hour**

---

## 4. Execution Logic

### 4.1 Signal Generation Pipeline
The system follows a strict waterfall logic for entry:

1.  **Filter 1 (Structure):** Is price inside an identified FVG or OTE zone?
2.  **Filter 2 (Regime):** Is ADX > 20 (Market is trending)?
3.  **Filter 3 (ML Probability):** Does Model `predict_proba` exceed 0.6?
4.  **Filter 4 (Risk):** Is the Reward-to-Risk ratio acceptable?

### 4.2 Exit Management
* **Stop Loss:** Dynamic Chandelier Exit (3.0x ATR).
* **Take Profit:** Fixed R:R or opposing structural liquidity pool.

---

## 5. Performance Report (Backtest)

**Warning:** Past performance is not indicative of future results.

### 5.1 Metrics Overview
* **Net Profit:** -$553 (Simulation)
* **Total Return:** -55.30%
* **Sharpe Ratio:** $-7.04$
* **Win Rate:** 9.84%
* **Total Trades:** 4867

### 5.2 Analysis
* **Profit Factor:** 0.06
* **Max Drawdown:** -55.3%

---

## 6. Technical Stack
The project is built using the following core technologies:

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn (SMOTE for imbalance)
* **Technical Analysis:** TA-Lib / Custom Pandas Implementation
* **Visualization:** Matplotlib, Seaborn

