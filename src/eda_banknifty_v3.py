import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from pandas.plotting import autocorrelation_plot
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'om_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda_banknifty_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(files)} files in om_data.")
    dfs = []
    
    # We only need date, time, and spot to reconstruct the BankNifty Index
    usecols = ['date', 'time', 'spot']
    
    for f in tqdm(files, desc="Loading BANKNIFTY SPOT"):
        df = pd.read_csv(f, usecols=usecols)
        df['date'] = df['date'].astype(str).str.replace('="', '').str.replace('"', '')
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])
    
    # Because options data has multiple rows per minute (different strikes),
    # the spot price is duplicated. We just take the first spot price for each minute.
    spot_1m = df.groupby('datetime')['spot'].first().to_frame(name='Close')
    spot_1m = spot_1m.sort_index()
    
    print(f"Reconstructed 1-Minute BankNifty Spot Index: {len(spot_1m):,} minutes of data.")
    return spot_1m

def layer1_microstructure_eda(df_1m):
    print("Running Layer 1: Market Structure EDA (1-Minute)...")
    out_dir = os.path.join(OUTPUT_DIR, 'layer1_intraday')
    os.makedirs(out_dir, exist_ok=True)
    
    df_1m['Returns'] = df_1m['Close'].pct_change()
    df_1m['time'] = df_1m.index.time
    df_1m['hour'] = df_1m.index.hour
    
    # 1. Intraday Volatility Profile
    intraday_vol = df_1m.groupby('time')['Returns'].std()
    plt.figure()
    intraday_vol.plot(color='crimson')
    plt.title("BankNifty Intraday Volatility Pattern (1-Min Returns Std Dev)")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_intraday_volatility_profile.png'), dpi=150)
    plt.close()
    
    # 2. Hourly Returns Distribution
    hourly_returns = df_1m.groupby('hour')['Returns'].mean()
    plt.figure()
    hourly_returns.plot(kind='bar', color='teal')
    plt.title("BankNifty Average Hourly Returns")
    plt.ylabel("Mean Return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_hourly_returns.png'), dpi=150)
    plt.close()
    
    # 3. Opening Gaps
    daily = df_1m.resample('D').agg({'Close': ['first', 'last']}).dropna()
    daily.columns = ['Open', 'Close']
    daily['Gap'] = (daily['Open'] - daily['Close'].shift(1)) / daily['Close'].shift(1)
    plt.figure()
    sns.histplot(daily['Gap'].dropna(), bins=50, color='orange', kde=True)
    plt.title("BankNifty Overnight Opening Gaps Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_opening_gaps.png'), dpi=150)
    plt.close()
    
    # 4. Day-wise Volatility Heatmap
    df_15m = df_1m.resample('15Min').last().dropna()
    df_15m['Date'] = df_15m.index.date
    df_15m['Time_Str'] = df_15m.index.strftime('%H:%M')
    df_15m['Returns_15m'] = df_15m['Close'].pct_change()
    
    pivot = df_15m.pivot_table(values='Returns_15m', index='Date', columns='Time_Str').dropna(how='all')
    plt.figure(figsize=(20, 8))
    sns.heatmap(pivot.abs(), cmap='Blues', cbar_kws={'label': 'Absolute Return'})
    plt.title("BankNifty 15-Min Volatility Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_volatility_heatmap.png'), dpi=150)
    plt.close()

def layer2_macro_eda(df_1m):
    print("Running Layer 2: Daily Aggregated EDA (Macro)...")
    out_dir = os.path.join(OUTPUT_DIR, 'layer2_daily')
    os.makedirs(out_dir, exist_ok=True)
    
    # Aggregate 1-minute Close prices to Daily OHLC
    df_daily = df_1m.resample('D').agg({
        'Close': ['first', 'max', 'min', 'last']
    }).dropna()
    df_daily.columns = ['Open', 'High', 'Low', 'Close']
    
    df_daily['Returns'] = df_daily['Close'].pct_change()
    
    # 1. Price & Moving Averages
    df_daily['MA20'] = df_daily['Close'].rolling(20).mean()
    df_daily['MA50'] = df_daily['Close'].rolling(50).mean()
    df_daily['MA200'] = df_daily['Close'].rolling(200).mean()
    
    plt.figure(figsize=(18,8))
    plt.plot(df_daily.index, df_daily['Close'], label='Close', color='black', alpha=0.7)
    plt.plot(df_daily.index, df_daily['MA20'], label='20 DMA', color='blue')
    plt.plot(df_daily.index, df_daily['MA50'], label='50 DMA', color='orange')
    plt.plot(df_daily.index, df_daily['MA200'], label='200 DMA', color='red')
    plt.legend()
    plt.title("BankNifty Daily Close & Moving Averages")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_moving_averages.png'), dpi=150)
    plt.close()
    
    # 2. Daily Returns Distribution
    plt.figure()
    sns.histplot(df_daily['Returns'].dropna(), bins=100, kde=True, color='navy')
    plt.title("BankNifty Daily Return Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_daily_returns_dist.png'), dpi=150)
    plt.close()
    
    # 3. Cumulative Returns
    df_daily['Cumulative_Return'] = (1 + df_daily['Returns'].fillna(0)).cumprod()
    plt.figure()
    plt.plot(df_daily.index, df_daily['Cumulative_Return'], color='green')
    plt.title("BankNifty Cumulative Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_cumulative_returns.png'), dpi=150)
    plt.close()
    
    # 4. Volatility (Annualized)
    df_daily['Volatility_20'] = df_daily['Returns'].rolling(20).std() * np.sqrt(252)
    plt.figure()
    plt.plot(df_daily.index, df_daily['Volatility_20'], color='purple')
    plt.title("BankNifty 20-Day Annualized Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_annualized_volatility.png'), dpi=150)
    plt.close()
    
    # 5. Drawdown
    rolling_max = df_daily['Close'].cummax()
    df_daily['Drawdown'] = (df_daily['Close'] - rolling_max) / rolling_max
    plt.figure()
    plt.fill_between(df_daily.index, df_daily['Drawdown'], color='red', alpha=0.5)
    plt.title(f"BankNifty Drawdown (Max: {round(df_daily['Drawdown'].min()*100,2)}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_drawdown.png'), dpi=150)
    plt.close()
    
    # 6. Monthly Returns Heatmap
    monthly_returns = df_daily['Close'].resample('M').last().pct_change().to_frame()
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    pivot_table = monthly_returns.pivot_table(values='Close', index='Year', columns='Month')
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot_table * 100, annot=True, fmt=".1f", cmap='RdYlGn', center=0)
    plt.title("BankNifty Monthly Returns Heatmap (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_monthly_returns_heatmap.png'), dpi=150)
    plt.close()
    
    # 7. Day of Week Analysis
    df_daily['Weekday'] = df_daily.index.day_name()
    weekday_returns = df_daily.groupby('Weekday')['Returns'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    plt.figure(figsize=(10,5))
    weekday_returns.plot(kind='bar', color='c')
    plt.title("BankNifty Average Return by Weekday")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '07_weekday_returns.png'), dpi=150)
    plt.close()
    
    # 8. Rolling Sharpe
    rolling_return = df_daily['Returns'].rolling(252).mean()
    rolling_std = df_daily['Returns'].rolling(252).std()
    df_daily['Rolling_Sharpe'] = (rolling_return / rolling_std) * np.sqrt(252)
    plt.figure()
    plt.plot(df_daily.index, df_daily['Rolling_Sharpe'], color='brown')
    plt.title("BankNifty 252-Day Rolling Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '08_rolling_sharpe.png'), dpi=150)
    plt.close()
    
    # 9. RSI
    delta = df_daily['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df_daily['RSI'] = 100 - (100 / (1 + rs))
    plt.figure()
    plt.plot(df_daily.index, df_daily['RSI'], color='indigo')
    plt.axhline(70, linestyle='--', color='red', alpha=0.5)
    plt.axhline(30, linestyle='--', color='green', alpha=0.5)
    plt.title("BankNifty 14-Day RSI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '09_rsi.png'), dpi=150)
    plt.close()
    
    # 10. Bollinger Bands
    df_daily['BB_Middle'] = df_daily['Close'].rolling(20).mean()
    std = df_daily['Close'].rolling(20).std()
    df_daily['BB_Upper'] = df_daily['BB_Middle'] + 2 * std
    df_daily['BB_Lower'] = df_daily['BB_Middle'] - 2 * std
    plt.figure(figsize=(16,7))
    plt.plot(df_daily.index, df_daily['Close'], label='Close', color='black')
    plt.plot(df_daily.index, df_daily['BB_Upper'], '--', label='Upper', color='red', alpha=0.5)
    plt.plot(df_daily.index, df_daily['BB_Lower'], '--', label='Lower', color='green', alpha=0.5)
    plt.fill_between(df_daily.index, df_daily['BB_Upper'], df_daily['BB_Lower'], color='gray', alpha=0.1)
    plt.legend()
    plt.title("BankNifty Bollinger Bands (20, 2)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '10_bollinger_bands.png'), dpi=150)
    plt.close()
    
    # Save Daily EDA CSV
    df_daily.to_csv(os.path.join(OUTPUT_DIR, "banknifty_daily_eda_output.csv"))
    print(f"Saved banknifty_daily_eda_output.csv to {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Starting Two-Layer BANKNIFTY EDA...")
    df_1m = load_data()
    layer1_microstructure_eda(df_1m)
    layer2_macro_eda(df_1m)
    print("BANKNIFTY EDA Complete!")
