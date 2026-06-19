import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_resample(data_dir):
    print("Loading FinNifty Data...")
    all_files = glob.glob(os.path.join(data_dir, "FINNIFTY_part*.csv"))
    skip_files = ['FINNIFTY_part1.csv', 'FINNIFTY_part2.csv']
    valid_files = [f for f in all_files if os.path.basename(f) not in skip_files]
    
    import re
    def extract_num(f):
        m = re.search(r'part(\d+)\.csv', f)
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
    
    print("Resampling to 5m bars...")
    df.set_index('datetime', inplace=True)
    
    resampled = df.resample('5min', origin='start_day').agg({
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
    
    # VWAP (Daily)
    df['date'] = df['datetime'].dt.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['typical_price'] * df['volume']
    
    # Cumulative sums per day
    grouped = df.groupby('date')
    df['cum_tp_v'] = grouped['tp_v'].cumsum()
    df['cum_volume'] = grouped['volume'].cumsum()
    
    df['vwap'] = df['cum_tp_v'] / df['cum_volume']
    
    # Z-Score
    df['zscore'] = (df['close'] - df['vwap']) / df['atr']
    
    return df.dropna().reset_index(drop=True)

def run_backtest(df):
    print("Running Backtest...")
    
    capital = 100000
    lot_size = 40  # FINNIFTY is 40
    brokerage_per_order = 20
    stt_rate = 0.0003
    slippage_pts = 0.5 # 0.5 points for Finnifty
    
    trades = []
    in_position = False
    
    df['prev_zscore'] = df['zscore'].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        time_obj = row['datetime'].time()
            
        if not in_position:
            # Trading Window: Strictly 10:30 to 13:30
            if time_obj < pd.to_datetime('10:30').time() or time_obj > pd.to_datetime('13:30').time():
                continue
                
            # Long Entry (price is below vwap significantly)
            if row['prev_zscore'] < -2.0:
                in_position = True
                entry_price = row['open'] + slippage_pts
                stop = entry_price - (0.50 * row['atr'])
                pos_type = 'LONG'
                entry_idx = i
                entry_time = row['datetime']
                entry_vwap = row['vwap']
                
            # Short Entry (price is above vwap significantly)
            elif row['prev_zscore'] > 2.0:
                in_position = True
                entry_price = row['open'] - slippage_pts
                stop = entry_price + (0.50 * row['atr'])
                pos_type = 'SHORT'
                entry_idx = i
                entry_time = row['datetime']
                entry_vwap = row['vwap']
                
        else:
            # Manage Position
            bars_held = i - entry_idx
            exit_price = None
            exit_reason = None
            
            high = row['high']
            low = row['low']
            close = row['close']
            vwap = row['vwap']
            
            if pos_type == 'LONG':
                if low <= stop:
                    exit_price = stop - slippage_pts
                    exit_reason = 'STOP'
                elif high >= vwap:
                    exit_price = vwap - slippage_pts
                    exit_reason = 'VWAP_TARGET'
                elif bars_held >= 9 or time_obj >= pd.to_datetime('15:15').time():
                    exit_price = close - slippage_pts
                    exit_reason = 'TIME'
                    
            elif pos_type == 'SHORT':
                if high >= stop:
                    exit_price = stop + slippage_pts
                    exit_reason = 'STOP'
                elif low <= vwap:
                    exit_price = vwap + slippage_pts
                    exit_reason = 'VWAP_TARGET'
                elif bars_held >= 9 or time_obj >= pd.to_datetime('15:15').time():
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
    
    print("\n--- Strategy 3: FinNifty Midday Mean Reversion ---")
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
    ax1.set_title('FinNifty Midday Reversion: Cumulative P&L')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.fill_between(trades_df['Exit_Time'], trades_df['Drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(trades_df['Exit_Time'], trades_df['Drawdown'], color='red')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('INR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy3_mr_pnl.png'))
    plt.close()

if __name__ == "__main__":
    os.makedirs("src/strategies", exist_ok=True)
    df = load_and_resample(r"data\raw\Equity_1min")
    df = compute_features(df)
    trades = run_backtest(df)
    generate_report(trades, "outputs/v2/strategy3_mr")
