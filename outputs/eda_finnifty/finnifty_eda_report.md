# FINNIFTY Exploratory Data Analysis Report

## 1. Dataset Overview
- **Total Records:** 404,367 minutes of data
- **Time Range:** 2021-08-04 10:00:00 to 2026-01-27 18:36:00

## 2. Key Findings

### Price & Volatility
- The price action exhibits multiple distinct regimes (trending vs ranging).
- Volatility (measured by ATR) spikes significantly during specific market periods, highlighting non-stationary volatility regimes.

### Returns Distribution
- **Mean 1-min Return:** 0.000001
- **Standard Deviation:** 0.000733
- **Skewness:** 0.13 (A negative skew means extreme down-moves are more likely than extreme up-moves).
- **Kurtosis:** 5341.29 (Normal distribution has kurtosis ~3. A value >> 3 means massive "fat tails", indicating extreme events happen much more frequently than a normal bell curve would predict).

### Intraday Seasonality
- The heatmap reveals structural market rhythms.
- The **Opening Hour (9:00 - 10:00)** consistently shows the highest volatility across all days.
- Volatility drops significantly during the mid-day "lunch" period.

## 3. Implications for the ML Model
1. **Fat Tails:** Linear models will fail. Tree-based models (like XGBoost) are required to capture the extreme, non-linear relationships.
2. **Time Features:** The `hour` and `day_of_week` features are highly predictive of market state and must be included in the model.
3. **Volatility Adjustment:** Static take-profits/stop-losses will perform poorly. Dynamic exits using ATR (like the Chandelier Exit) are strongly validated by the changing volatility regimes.

*(Please see the .png charts in this folder for visualizations)*
