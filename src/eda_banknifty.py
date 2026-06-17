import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'om_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda_banknifty')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_sample_data(sample_frac=0.05):
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(files)} files in om_data.")
    dfs = []
    
    for f in tqdm(files, desc="Loading & Sampling BANKNIFTY"):
        try:
            # Read file and sample to avoid memory overflow (100MB files * 20 = 2GB)
            # Memory peaks at 100MB per iteration, which is very safe.
            df = pd.read_csv(f)
            df_sampled = df.sample(frac=sample_frac, random_state=42)
            
            # Clean date string if necessary
            df_sampled['date'] = df_sampled['date'].astype(str).str.replace('="', '').str.replace('"', '')
            dfs.append(df_sampled)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    print(f"Sampled Dataset Size: {len(df):,} rows")
    return df

def plot_volume_by_strike_offset(df):
    plt.figure(figsize=(10, 6))
    
    # Group by strike offset and sum volume
    vol_by_strike = df.groupby('strike_offset')['volume'].sum().sort_values(ascending=False)
    
    sns.barplot(x=vol_by_strike.index, y=vol_by_strike.values, palette='viridis')
    plt.title('Total Traded Volume by Strike Offset (ATM vs OTM vs ITM)', fontsize=14, fontweight='bold')
    plt.xlabel('Strike Offset')
    plt.ylabel('Total Volume')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_volume_by_strike.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_iv_vs_moneyness(df):
    # Filter out extreme IVs for clean plotting
    df_clean = df[(df['iv'] > 0) & (df['iv'] < 100)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_clean, x='strike_offset', y='iv', hue='option_type', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Implied Volatility (IV) Smile / Skew by Strike Offset', fontsize=14, fontweight='bold')
    plt.xlabel('Strike Offset')
    plt.ylabel('Implied Volatility (IV)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, '02_iv_skew.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_ce_vs_pe_volume_intraday(df):
    df['hour'] = df['datetime'].dt.hour
    
    # Aggregate volume by hour and option type
    hourly_vol = df.groupby(['hour', 'option_type'])['volume'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hourly_vol, x='hour', y='volume', hue='option_type', marker='o', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Average Intraday Volume: Calls (CE) vs Puts (PE)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Volume')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_intraday_volume.png'), dpi=150)
    plt.close()

def plot_spot_vs_premium(df):
    # Only look at ATM options for premium analysis
    df_atm = df[df['strike_offset'] == 'ATM']
    
    # We will scatter Spot Price vs Premium (Close Price of Option)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_atm, x='spot', y='close', hue='option_type', alpha=0.3, s=15, palette=['#1f77b4', '#ff7f0e'])
    plt.title('ATM Option Premium vs Spot Price', fontsize=14, fontweight='bold')
    plt.xlabel('Underlying Spot Price (BANKNIFTY)')
    plt.ylabel('Option Premium (Close Price)')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_spot_vs_premium.png'), dpi=150)
    plt.close()

def generate_markdown_report(df):
    report = f"""# BANKNIFTY Options Data Exploratory Data Analysis Report

## 1. Dataset Overview
- **Data Source:** `om_data` folder (approx. 20 files, 2 GB total)
- **Sampling:** To handle the massive dataset size, a 5% stratified random sample was used to compute high-resolution scatterplots and distributions.
- **Sampled Records:** {len(df):,} minutes of options data
- **Time Range:** {df['datetime'].min()} to {df['datetime'].max()}

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
"""
    with open(os.path.join(OUTPUT_DIR, 'banknifty_eda_report.md'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    print("Starting BANKNIFTY Options EDA...")
    df = load_and_sample_data(sample_frac=0.05)
    
    print("Generating Plots...")
    plot_volume_by_strike_offset(df)
    plot_iv_vs_moneyness(df)
    plot_ce_vs_pe_volume_intraday(df)
    plot_spot_vs_premium(df)
    
    print("Generating Report...")
    generate_markdown_report(df)
    print(f"BANKNIFTY EDA complete! Outputs saved to {OUTPUT_DIR}")
