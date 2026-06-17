# BANKNIFTY Options Data Exploratory Data Analysis Report

## 1. Dataset Overview
- **Data Source:** `om_data` folder (approx. 20 files, 2 GB total)
- **Sampling:** To handle the massive dataset size, a 5% stratified random sample was used to compute high-resolution scatterplots and distributions.
- **Sampled Records:** 966,489 minutes of options data
- **Time Range:** 2021-08-04 10:00:00 to 2026-06-09 15:28:00

## 2. Key Findings

### Liquidity & Volume Profile
- **Strike Offsets:** The vast majority of trading volume is heavily concentrated around At-The-Money (ATM) strikes. Volume drops exponentially as you move into Deep OTM (Out of The Money) or Deep ITM (In The Money) strikes. 
- **Intraday Flow:** Volume follows a classic "U-shape" curve during the day. High liquidity at the open (9:00 - 10:00), a lull during the mid-day session, and a massive spike in the final hour (14:00 - 15:30) as positions are squared off.

### Volatility Skew & Smile
- The Implied Volatility (IV) distribution shows a distinct structure across strikes.
- Out-of-the-money (OTM) Puts typically price in a higher IV premium compared to equidistant OTM Calls. This phenomenon (Volatility Skew) represents the market pricing in crash-risk / downside tail-risk heavily.

### Call vs Put Dynamics
- Both Call (CE) and Put (PE) options show symmetric premium expansion based on Spot movements, but Puts generally retain a slight IV premium.
- Intraday volume between Calls and Puts is relatively balanced, though specific trend days will skew the ratio significantly.

## 3. Implications for the ML Model
1. **Feature Engineering:** If the trading system is trading the Spot index (FINNIFTY/BANKNIFTY), the `IV` and `OI` data from the options chain can act as leading indicators. Adding the `Put-Call Ratio (PCR)` or `ATM IV` as predictive features to the XGBoost model could yield high alpha.
2. **Execution Slippage:** Because liquidity drops sharply outside of ATM, any algorithm must ensure it enters trades either in Spot/Futures, or strictly stays near ATM strikes if trading the options directly, to avoid massive bid-ask slippage.

*(Please see the .png charts in this folder for visualizations)*
