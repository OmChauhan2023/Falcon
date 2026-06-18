# TEAM FALCON: VALIDATION GAUNTLET REPORT

> Goal: Relentlessly attack the trading system to uncover hidden curve-fitting or structural fragilities.

## Initialization: Training Baseline Engine
Baseline model trained.

### Global Test Set (Baseline Filtered)
- **Trades:** 115 | **Win Rate:** 73.9% | **PF:** 3.17 | **Sharpe:** 10.34
- **Expectancy per Trade:** 1.37R | **Return:** 28.74%

---

## 1. Segment Decomposition
Evaluating stability across the 3 independent test segments.

- **Segment 2:** Trades: 115 | PF:  3.17 | Sharpe: 10.34 | Expectancy:  1.37R | Return: 28.74%

---

## 2. Temporal Stability
Evaluating chronological decay across the test set (First 33%, Middle 33%, Last 33%).

- **First 33%:** Trades: 45  | PF:  2.78 | Sharpe: 10.74 | Expectancy:  1.34R
- **Middle 33%:** Trades: 38  | PF:  3.18 | Sharpe: 10.14 | Expectancy:  1.29R
- **Last 33%:** Trades: 32  | PF:  3.70 | Sharpe: 10.39 | Expectancy:  1.49R

---

## 3. Threshold Stability
Testing classification boundaries to detect curve-fitting.

| Threshold | Trades | Win Rate | Expectancy | PF | Sharpe | Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.60 | 410 | 62.9% | 1.15R | 2.59 | 16.12 | 117.69% |
| 0.62 | 252 | 69.4% | 1.33R | 3.39 | 15.61 | 83.64% |
| 0.64 | 151 | 73.5% | 1.43R | 4.08 | 13.25 | 52.99% |
| 0.65 | 115 | 73.9% | 1.37R | 3.17 | 10.34 | 28.74% |
| 0.66 | 88 | 76.1% | 1.47R | 3.82 | 10.21 | 26.93% |
| 0.68 | 40 | 80.0% | 1.54R | 4.78 | 7.37 | 12.65% |
| 0.70 | 19 | 78.9% | 1.56R | 4.50 | 4.89 | 5.28% |
| 0.72 | 7 | 71.4% | 1.35R | 2.49 | 2.19 | 0.91% |

---

## 4. Slippage Stress Matrix
Evaluating survival against extreme liquidity friction.

| Slippage (pts) | Trades | Win Rate | Expectancy | PF | Sharpe | Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.0 | 115 | 73.9% | 1.37R | 3.17 | 10.34 | 28.74% |
| 2.0 | 115 | 72.2% | 1.25R | 2.73 | 9.02 | 24.58% |
| 3.0 | 115 | 70.4% | 1.14R | 2.35 | 7.61 | 20.53% |
| 4.0 | 115 | 70.4% | 1.03R | 2.05 | 6.33 | 16.98% |
| 5.0 | 115 | 66.1% | 0.92R | 1.79 | 5.04 | 13.50% |

---

## 5. Trade Concentration
Checking dependency on outlier lottery-ticket trades.

| Excluded Outliers | Removed PnL | Remaining PnL | Retained % | Profitable? |
| :--- | :--- | :--- | :--- | :--- |
| Top 1% | Rs33,901 | Rs253,478 | 88.2% | Yes |
| Top 5% | Rs87,993 | Rs199,386 | 69.4% | Yes |
| Top 10% | Rs148,752 | Rs138,627 | 48.2% | Yes |
