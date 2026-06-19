import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_resample(data_dir):
    print("Loading BankNifty Data...")
    all_files = glob.glob(os.path.join(data_dir, "BANKNIFTY_part_*.csv"))
    skip_files = ['BANKNIFTY_part_1.csv', 'BANKNIFTY_part_2.csv']
    valid_files = [f for f in all_files if os.path.basename(f) not in skip_files]
    
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
    df['date'] = df['date'].astype(str).str.replace('="', '', regex=False).str.replace('"', '', regex=False)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    print("Resampling to 15m bars...")
    df.set_index('datetime', inplace=True)
    
    resampled = df.resample('15min', origin='start_day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled = resampled.reset_index()
    resampled = resampled[
        (resampled['datetime'].dt.time >= pd.to_datetime('09:15').time()) &
        (resampled['datetime'].dt.time <= pd.to_datetime('15:30').time())
    ]
    return resampled.reset_index(drop=True)

def compute_features(df):
    print("Computing features...")
    # ATR (14 period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = np.max(ranges, axis=1).rolling(14).mean()
    
    # Bollinger Bands Squeeze
    df['ma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['ma_20'] + (2 * df['std_20'])
    df['lower_band'] = df['ma_20'] - (2 * df['std_20'])
    df['bb_width'] = df['upper_band'] - df['lower_band']
    # Rank BB width over last 252 bars (approx 10 days of 15m bars)
    df['bb_squeeze_pct'] = df['bb_width'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    
    # Volume Ratio
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['vol_ma_20'].shift(1)
    
    # ORB High / Low
    # First 2 candles are 09:15 and 09:30
    df['date'] = df['datetime'].dt.date
    # Find the high and low of the first two bars of each day
    # Group by date, take head(2), then find max/min
    orb = df.groupby('date').apply(lambda x: pd.Series({
        'orb_high': x.head(2)['high'].max(),
        'orb_low': x.head(2)['low'].min()
    })).reset_index()
    
    df = df.merge(orb, on='date', how='left')
    return df.dropna().reset_index(drop=True)

def run_backtest(df):
    print("Running Backtest...")
    
    capital = 100000
    lot_size = 15
    brokerage_per_order = 20
    stt_rate = 0.0003
    slippage_pts = 1.0
    
    trades = []
    in_position = False
    
    # Shift features for execution safely
    df['prev_close'] = df['close'].shift(1)
    df['prev_vol_ratio'] = df['volume_ratio'].shift(1)
    df['prev_squeeze'] = df['bb_squeeze_pct'].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Only trade after 09:45 (after ORB is formed)
        time_obj = row['datetime'].time()
        if time_obj <= pd.to_datetime('09:45').time():
            continue
            
        if not in_position:
            # We can enter up until 14:15 to allow for a 1-hour hold before 15:15
            if time_obj >= pd.to_datetime('14:15').time():
                continue
                
            # Filters
            if row['prev_squeeze'] < 0.30:
                continue
            if row['prev_vol_ratio'] < 1.5:
                continue
                
            # Long Entry
            if row['prev_close'] > row['orb_high']:
                in_position = True
                entry_price = row['open'] + slippage_pts
                target = entry_price + (0.75 * row['atr'])
                stop = entry_price - (0.50 * row['atr'])
                pos_type = 'LONG'
                entry_idx = i
                entry_time = row['datetime']
                
            # Short Entry
            elif row['prev_close'] < row['orb_low']:
                in_position = True
                entry_price = row['open'] - slippage_pts
                target = entry_price - (0.75 * row['atr'])
                stop = entry_price + (0.50 * row['atr'])
                pos_type = 'SHORT'
                entry_idx = i
                entry_time = row['datetime']
                
        else:
            # Manage Position
            bars_held = i - entry_idx
            exit_price = None
            exit_reason = None
            
            high = row['high']
            low = row['low']
            close = row['close']
            
            if pos_type == 'LONG':
                if low <= stop:
                    exit_price = stop - slippage_pts
                    exit_reason = 'STOP'
                elif high >= target:
                    exit_price = target - slippage_pts # slippage applied to target too
                    exit_reason = 'TARGET'
                elif bars_held >= 4 or time_obj >= pd.to_datetime('15:15').time():
                    exit_price = close - slippage_pts
                    exit_reason = 'TIME'
                    
            elif pos_type == 'SHORT':
                if high >= stop:
                    exit_price = stop + slippage_pts
                    exit_reason = 'STOP'
                elif low <= target:
                    exit_price = target + slippage_pts
                    exit_reason = 'TARGET'
                elif bars_held >= 4 or time_obj >= pd.to_datetime('15:15').time():
                    exit_price = close + slippage_pts
                    exit_reason = 'TIME'
                    
            if exit_price is not None:
                in_position = False
                
                # PnL Calc
                gross_pts = (exit_price - entry_price) if pos_type == 'LONG' else (entry_price - exit_price)
                gross_pnl = gross_pts * lot_size
                
                turnover = (entry_price + exit_price) * lot_size
                stt = turnover * stt_rate
                brokerage = brokerage_per_order * 2
                net_pnl = gross_pnl - stt - brokerage
                
                trades.append({
                    'Entry_Time': entry_time,
                    'Exit_Time': row['datetime'],
                    'Type': pos_type,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Reason': exit_reason,
                    'Gross_Pts': gross_pts,
                    'Net_PnL': net_pnl,
                    'Bars_Held': bars_held
                })
                
    return pd.DataFrame(trades)

def generate_report(trades_df, output_dir):
    print("Generating Report...")
    os.makedirs(output_dir, exist_ok=True)
    
    if len(trades_df) == 0:
        print("No trades executed.")
        return
        
    trades_df['Cumulative_PnL'] = trades_df['Net_PnL'].cumsum()
    trades_df['Peak'] = trades_df['Cumulative_PnL'].cummax()
    trades_df['Drawdown'] = trades_df['Cumulative_PnL'] - trades_df['Peak']
    
    win_rate = (trades_df['Net_PnL'] > 0).mean() * 100
    avg_win = trades_df[trades_df['Net_PnL'] > 0]['Net_PnL'].mean()
    avg_loss = abs(trades_df[trades_df['Net_PnL'] <= 0]['Net_PnL'].mean())
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    total_pnl = trades_df['Cumulative_PnL'].iloc[-1]
    max_dd = trades_df['Drawdown'].min()
    
    print("\n--- Strategy 1: BankNifty ORB Expansion ---")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Avg Win:      Rs{avg_win:.2f}")
    print(f"Avg Loss:     Rs{avg_loss:.2f}")
    print(f"Expectancy:   Rs{expectancy:.2f} per trade")
    print(f"Total PnL:    Rs{total_pnl:.2f}")
    print(f"Max Drawdown: Rs{max_dd:.2f}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(trades_df['Exit_Time'], trades_df['Cumulative_PnL'], color='blue', label='Cumulative Net PnL')
    ax1.set_title('BankNifty ORB Expansion: Cumulative P&L')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.fill_between(trades_df['Exit_Time'], trades_df['Drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(trades_df['Exit_Time'], trades_df['Drawdown'], color='red')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('INR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy1_orb_pnl.png'))
    plt.close()

if __name__ == "__main__":
    os.makedirs("src/strategies", exist_ok=True)
    df = load_and_resample("om_data")
    df = compute_features(df)
    trades = run_backtest(df)
    generate_report(trades, "outputs/v2/strategy1_orb")
