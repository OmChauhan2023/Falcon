# TEAM FALCON: WALK-FORWARD VALIDATION REPORT

> Goal: Prove the strategy survives out-of-sample across an expanding window while benchmarking against naive baselines.

## 1. Global Data Initialization
Total Segments identified: 33
Walk-Forward Setup: Train Window starts at 5 segments, Expanding sequentially.

## 2. Walk-Forward Execution Log

| Fold | Train PF | Test PF | Gen Gap | Rule PF | Falcon Win? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.15 | 0.06 | -0.09 | 0.07 | No |
| 2 | 0.10 | 0.14 | +0.04 | 0.05 | Yes |
| 3 | 0.13 | 0.03 | -0.11 | 0.12 | No |
| 4 | 0.15 | 0.14 | -0.01 | 0.13 | Yes |
| 5 | 0.13 | 0.32 | +0.19 | 0.12 | Yes |
| 6 | 0.21 | 0.22 | +0.00 | 0.14 | Yes |
| 7 | 0.21 | 0.04 | -0.17 | 0.09 | No |
| 8 | 0.23 | 0.39 | +0.16 | 0.17 | Yes |
| 9 | 0.29 | 0.29 | -0.01 | 0.26 | Yes |
| 10 | 0.29 | 0.04 | -0.25 | 0.16 | No |
| 11 | 0.28 | 0.16 | -0.11 | 0.14 | Yes |
| 12 | 0.26 | 0.02 | -0.24 | 0.09 | No |
| 13 | 0.22 | 0.00 | -0.22 | 0.21 | No |
| 14 | 0.32 | 0.22 | -0.10 | 0.15 | Yes |
| 15 | 0.50 | 0.03 | -0.48 | 0.15 | No |
| 16 | 0.28 | 0.04 | -0.24 | 0.11 | No |
| 17 | 0.35 | 0.04 | -0.31 | 0.18 | No |
| 18 | 0.25 | 0.03 | -0.22 | 0.10 | No |
| 19 | 0.29 | 0.15 | -0.13 | 0.26 | No |
| 20 | 0.36 | 0.22 | -0.14 | 0.19 | Yes |
| 21 | 0.27 | 0.03 | -0.24 | 0.14 | No |
| 22 | 0.38 | 0.00 | -0.38 | 0.08 | No |
| 23 | 0.48 | 0.00 | -0.48 | 0.12 | No |
| 24 | 0.37 | 0.00 | -0.37 | 0.07 | No |
| 25 | 0.39 | 0.08 | -0.31 | 0.16 | No |
| 26 | 0.39 | 0.00 | -0.39 | 0.09 | No |
| 27 | 0.52 | 0.02 | -0.50 | 0.08 | No |
| 28 | 0.52 | 0.00 | -0.52 | 0.09 | No |

**Fold Dominance:** Falcon beat the Rule Baseline in 9 out of 28 folds.

## 3. Cumulative Out-Of-Sample Results

| Model Benchmark | Trades | Win Rate | Expectancy | PF | Sharpe | Return | Total PnL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
