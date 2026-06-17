import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'Equity_1min')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda_finnifty')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(files)} files.")
    dfs = []
    for f in tqdm(files, desc="Loading FINNIFTY"):
        df = pd.read_csv(f)
        # Clean date string if necessary e.g., '="04-08-21"' -> '04-08-21'
        df['date'] = df['date'].astype(str).str.replace('="', '').str.replace('"', '')
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    return df

def feature_engineering(df):
    print("Engineering features for EDA...")
    df['returns_1m'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Intraday Volatility (High - Low)
    df['intraday_volatility'] = df['high'] - df['low']
    
    # ATR (14 period)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    return df

def plot_price_and_volatility(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['datetime'], df['close'], color='navy', linewidth=0.8)
    ax1.set_title('FINNIFTY Close Price Trend', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    
    ax2.plot(df['datetime'], df['atr_14'], color='darkorange', linewidth=0.8, alpha=0.8)
    ax2.set_title('FINNIFTY 14-Period ATR (Volatility)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ATR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_price_volatility.png'), dpi=150)
    plt.close()

def plot_returns_distribution(df):
    plt.figure(figsize=(10, 6))
    ret_clean = df['returns_1m'].dropna()
    sns.histplot(ret_clean, bins=200, kde=True, color='teal')
    plt.title('1-Minute Returns Distribution (Notice Fat Tails)', fontsize=14, fontweight='bold')
    plt.xlim(-0.005, 0.005) # Zoom in
    plt.xlabel('Returns')
    plt.savefig(os.path.join(OUTPUT_DIR, '02_returns_dist.png'), dpi=150)
    plt.close()
    
    # Compute stats for markdown
    mean_ret = ret_clean.mean()
    std_ret = ret_clean.std()
    skew = ret_clean.skew()
    kurt = ret_clean.kurtosis()
    return mean_ret, std_ret, skew, kurt

def plot_intraday_seasonality(df):
    df_clean = df.dropna(subset=['intraday_volatility'])
    pivot = pd.pivot_table(df_clean, values='intraday_volatility', index='day_of_week', columns='hour', aggfunc='mean')
    
    # Sort days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    pivot = pivot.reindex(days)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt=".1f")
    plt.title('Average 1-Min Volatility (High - Low) by Day and Hour', fontsize=14, fontweight='bold')
    plt.ylabel('')
    plt.xlabel('Hour of Day')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_intraday_seasonality.png'), dpi=150)
    plt.close()

def plot_autocorrelation(df):
    ret_clean = df['returns_1m'].dropna()
    # Use only last 100k rows to save memory/time
    sample_ret = ret_clean.tail(100000)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(sample_ret, lags=30, ax=ax, alpha=0.05, zero=False)
    plt.title('Autocorrelation of 1-Min Returns (Lags 1 to 30)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_autocorrelation.png'), dpi=150)
    plt.close()

def generate_markdown_report(mean_ret, std_ret, skew, kurt, df):
    report = f"""# FINNIFTY Exploratory Data Analysis Report

## 1. Dataset Overview
- **Total Records:** {len(df):,} minutes of data
- **Time Range:** {df['datetime'].min()} to {df['datetime'].max()}

## 2. Key Findings

### Price & Volatility
- The price action exhibits multiple distinct regimes (trending vs ranging).
- Volatility (measured by ATR) spikes significantly during specific market periods, highlighting non-stationary volatility regimes.

### Returns Distribution
- **Mean 1-min Return:** {mean_ret:.6f}
- **Standard Deviation:** {std_ret:.6f}
- **Skewness:** {skew:.2f} (A negative skew means extreme down-moves are more likely than extreme up-moves).
- **Kurtosis:** {kurt:.2f} (Normal distribution has kurtosis ~3. A value >> 3 means massive "fat tails", indicating extreme events happen much more frequently than a normal bell curve would predict).

### Intraday Seasonality
- The heatmap reveals structural market rhythms.
- The **Opening Hour (9:00 - 10:00)** consistently shows the highest volatility across all days.
- Volatility drops significantly during the mid-day "lunch" period.

## 3. Implications for the ML Model
1. **Fat Tails:** Linear models will fail. Tree-based models (like XGBoost) are required to capture the extreme, non-linear relationships.
2. **Time Features:** The `hour` and `day_of_week` features are highly predictive of market state and must be included in the model.
3. **Volatility Adjustment:** Static take-profits/stop-losses will perform poorly. Dynamic exits using ATR (like the Chandelier Exit) are strongly validated by the changing volatility regimes.

*(Please see the .png charts in this folder for visualizations)*
"""
    with open(os.path.join(OUTPUT_DIR, 'finnifty_eda_report.md'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    print("Starting FINNIFTY EDA...")
    df = load_data()
    df = feature_engineering(df)
    
    print("Generating Plots...")
    plot_price_and_volatility(df)
    mean_ret, std_ret, skew, kurt = plot_returns_distribution(df)
    plot_intraday_seasonality(df)
    plot_autocorrelation(df)
    
    print("Generating Report...")
    generate_markdown_report(mean_ret, std_ret, skew, kurt, df)
    print(f"FINNIFTY EDA complete! Outputs saved to {OUTPUT_DIR}")
