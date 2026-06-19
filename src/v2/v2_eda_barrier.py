import os
import glob
import pandas as pd
import numpy as np

def load_data(data_dir):
    print("Loading BankNifty Data...")
    all_files = glob.glob(os.path.join(data_dir, "BANKNIFTY_part_*.csv"))
    # Skip part_1 and part_2 to avoid the massive 6-month gap identified in V1
    skip_files = ['BANKNIFTY_part_1.csv', 'BANKNIFTY_part_2.csv']
    valid_files = [f for f in all_files if os.path.basename(f) not in skip_files]
    
    # Sort files naturally
    import re
    def extract_num(f):
        m = re.search(r'part_(\d+)\.csv', f)
        return int(m.group(1)) if m else 0
    valid_files.sort(key=extract_num)
    
    dfs = []
    for f in valid_files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower()
    
    # Clean excel formatting in dates like ="31-01-22"
    df['date'] = df['date'].astype(str).str.replace('="', '', regex=False).str.replace('"', '', regex=False)
    
    # Parse exactly using format
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def resample_to_15m(df):
    print("Resampling to 15m bars...")
    df.set_index('datetime', inplace=True)
    
    # Resample logic keeping market boundaries intact
    # 15m rule: 9:15-9:30, 9:30-9:45, etc.
    # pandas resample('15Min') starting at 9:15
    resampled = df.resample('15min', origin='start_day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled = resampled.reset_index()
    # Filter to only keep regular market hours 09:15 to 15:30
    resampled = resampled[
        (resampled['datetime'].dt.time >= pd.to_datetime('09:15').time()) &
        (resampled['datetime'].dt.time <= pd.to_datetime('15:30').time())
    ]
    return resampled.reset_index(drop=True)

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def compute_excursions(df, holding_bars):
    """
    Computes Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE)
    normalized by ATR at entry for Long positions.
    (For shorts, MFE/MAE are symmetric in magnitude).
    """
    # Create rolling windows for future highs and lows
    # Shift(-1) so we look at the NEXT 'holding_bars' bars
    future_highs = df['high'].shift(-1).rolling(holding_bars).max().shift(-(holding_bars-1))
    future_lows = df['low'].shift(-1).rolling(holding_bars).min().shift(-(holding_bars-1))
    
    # MFE (Long): Highest high in future window - Entry Close
    mfe_pts = future_highs - df['close']
    
    # MAE (Long): Entry Close - Lowest low in future window
    mae_pts = df['close'] - future_lows
    
    # Normalize by ATR
    mfe_atr = mfe_pts / df['atr']
    mae_atr = mae_pts / df['atr']
    
    return mfe_atr, mae_atr

def print_percentiles(mfe, mae, name):
    print(f"\n--- {name} Distributions ---")
    p_levels = [0.5, 0.75, 0.90, 0.95]
    print(f"{'Percentile':<12} | {'MFE (ATR)':<12} | {'MAE (ATR)':<12}")
    print("-" * 42)
    for p in p_levels:
        m_val = mfe.quantile(p)
        a_val = mae.quantile(p)
        print(f"{int(p*100):<2}%         | {m_val:<12.2f} | {a_val:<12.2f}")

def run_label_balance_grid(df, mfe_4, mae_4, mfe_8, mae_8):
    print("\n--- Label Balance Grid ---")
    
    # Define grid to test
    tps = [0.5, 0.75, 1.0, 1.25, 1.5]
    sls = [0.25, 0.5, 0.75, 1.0, 1.25]
    
    results = []
    
    # Evaluate 4-bar hold
    for tp in tps:
        for sl in sls:
            # Positive label condition:
            # MFE >= TP AND MAE < SL
            # Meaning it hits Target BEFORE it hits Stop in the window
            # (Note: This is an approximation since we don't know the exact intra-bar path 
            # within the future window, but over 4-8 bars, MFE >= TP and MAE < SL is a very strong proxy)
            
            # Actually, triple barrier is exactly:
            # Hit target first -> 1
            # Hit stop first -> -1
            # Hit neither -> 0
            # A strict approximation:
            # If MFE >= TP and MAE < SL -> Hit target first (100% sure)
            # If MAE >= SL and MFE < TP -> Hit stop first (100% sure)
            # If BOTH MFE >= TP and MAE >= SL -> We don't know which hit first. We can assume the worst (hit stop first) or ignore.
            # Let's count "Strict Positive" as MFE >= TP & MAE < SL
            pos_mask_4 = (mfe_4 >= tp) & (mae_4 < sl)
            pos_rate_4 = pos_mask_4.mean() * 100
            
            pos_mask_8 = (mfe_8 >= tp) & (mae_8 < sl)
            pos_rate_8 = pos_mask_8.mean() * 100
            
            results.append({
                'TP (ATR)': tp,
                'SL (ATR)': sl,
                '4-Bar Pos Rate': f"{pos_rate_4:.1f}%",
                '8-Bar Pos Rate': f"{pos_rate_8:.1f}%"
            })
            
    grid_df = pd.DataFrame(results)
    print(grid_df.to_string(index=False))

if __name__ == "__main__":
    os.makedirs("outputs/v2", exist_ok=True)
    raw_df = load_data("om_data")
    df = resample_to_15m(raw_df)
    
    df['atr'] = compute_atr(df, 14)
    df.dropna(subset=['atr'], inplace=True)
    
    # Calculate for 4 bars (1 hour) and 8 bars (2 hours)
    mfe_4, mae_4 = compute_excursions(df, holding_bars=4)
    mfe_8, mae_8 = compute_excursions(df, holding_bars=8)
    
    print_percentiles(mfe_4, mae_4, "4-Bar (1 Hour) Hold")
    print_percentiles(mfe_8, mae_8, "8-Bar (2 Hour) Hold")
    
    run_label_balance_grid(df, mfe_4, mae_4, mfe_8, mae_8)
