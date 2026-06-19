import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from finnifty_trading_system import DataLoader, FeatureEngineer, DatasetPreparer, XGBoostTrainer, TradingBacktester

REPORT_FILE = os.path.join(BASE_DIR, "outputs", "reports", "walk_forward_report.md")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calc_expectancy(trades_df):
    if len(trades_df) == 0: return 0.0
    wins = trades_df[trades_df['R'] > 0]
    losses = trades_df[trades_df['R'] <= 0]
    win_rate = len(wins) / len(trades_df)
    avg_win = wins['R'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['R'].mean()) if len(losses) > 0 else 0
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

def get_cumulative_metrics(trades_df, start_capital=1000000):
    if len(trades_df) == 0:
        return 0, 0, 0, 0, 0, 0, 0
        
    wins = trades_df[trades_df['Net_PnL'] > 0]
    losses = trades_df[trades_df['Net_PnL'] < 0]
    win_rate = len(wins) / len(trades_df) * 100
    pf = wins['Net_PnL'].sum() / abs(losses['Net_PnL'].sum()) if abs(losses['Net_PnL'].sum()) > 0 else float('inf')
    avg_r = trades_df['R'].mean()
    expectancy = calc_expectancy(trades_df)
    
    total_pnl = trades_df['Net_PnL'].sum()
    total_ret = (total_pnl / start_capital) * 100
    
    # Calculate Sharpe
    # Simulate an equity curve from daily aggregate PnL to compute proper Sharpe
    # If trades span multiple days, we group by date
    trades_df['Date'] = pd.to_datetime(trades_df['Entry_Time']).dt.date
    daily_pnl = trades_df.groupby('Date')['Net_PnL'].sum()
    daily_returns = daily_pnl / start_capital # Approx daily return
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    return len(trades_df), win_rate, pf, sharpe, avg_r, total_ret, expectancy

def get_fold_metrics(bt, X_test, test_meta, sigs, probs):
    trades_df, _ = bt.backtest(X_test, test_meta, sigs, probs)
    if len(trades_df) == 0:
        return 0, 0, 0
    wins = len(trades_df[trades_df['Net_PnL'] > 0])
    wr = wins / len(trades_df) * 100
    exp = calc_expectancy(trades_df)
    pnl = trades_df['Net_PnL'].sum()
    return len(trades_df), wr, exp, pnl, trades_df

def run_walk_forward():
    sys.stdout = Logger(REPORT_FILE)
    
    print("# TEAM FALCON: WALK-FORWARD VALIDATION REPORT\n")
    print("> Goal: Prove the strategy survives out-of-sample across an expanding window while benchmarking against naive baselines.\n")

    print("## 1. Global Data Initialization")
    loader = DataLoader()
    raw_df = loader.load_data() # Loads all CSV files
    
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(raw_df)
    
    # Overwrite segment with Month for proper walk-forward boundaries
    df['segment'] = df['datetime'].dt.to_period('M').astype(str)
    
    preparer = DatasetPreparer()
    feat_cols = DatasetPreparer.get_feature_columns(False)
    X, y, meta, _ = preparer.prepare_ml_dataset(df, feat_cols)
    
    segments = sorted(meta['segment'].unique())
    print(f"Total Segments identified: {len(segments)}")
    
    min_train_segs = 5
    if len(segments) <= min_train_segs:
        print("Not enough segments for Walk-Forward.")
        return
        
    print(f"Walk-Forward Setup: Train Window starts at {min_train_segs} segments, Expanding sequentially.\n")
    
    # Accumulators for out-of-sample trades
    cumulative_trades = {
        'Falcon': [],
        'Rule_TWAP': [],
        'Random': []
    }
    
    falcon_wins = 0
    total_folds = 0
    
    print("## 2. Walk-Forward Execution Log\n")
    print("| Fold | Train PF | Test PF | Gen Gap | Rule PF | Falcon Win? |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for i in range(min_train_segs, len(segments)):
        test_seg = segments[i]
        train_segs = segments[:i]
        
        train_mask = meta['segment'].isin(train_segs)
        test_mask = meta['segment'] == test_seg
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_train = meta[train_mask].reset_index(drop=True)
        meta_test = meta[test_mask].reset_index(drop=True)
        
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        total_folds += 1
        
        # Train Model
        trainer = XGBoostTrainer()
        from sklearn.model_selection import train_test_split
        X_tr, X_v, y_tr, y_v = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
        
        X_tr_s, X_v_s, X_te_s = trainer.scale_features(X_tr, X_v, X_test)
        # Also need to scale full train set for Train PF eval
        X_train_s = trainer.scaler.transform(X_train)
        
        model = trainer.train_model(X_tr_s, y_tr, X_v_s, y_v, verbose=False)
        
        # 1. Falcon (Full Model + Filter)
        bt_falcon = TradingBacktester(exp_skip_low_vol=True)
        bt_falcon.threshold = 0.65
        bt_falcon.model, bt_falcon.scaler = model, trainer.scaler
        
        # Train PF
        sigs_tr, probs_tr = bt_falcon.generate_signals(X_train, meta_train)
        _, _, _, _, tr_f_train = get_fold_metrics(bt_falcon, X_train, meta_train, sigs_tr, probs_tr)
        train_pf = tr_f_train['Net_PnL'][tr_f_train['Net_PnL'] > 0].sum() / abs(tr_f_train['Net_PnL'][tr_f_train['Net_PnL'] < 0].sum()) if len(tr_f_train) > 0 and abs(tr_f_train['Net_PnL'][tr_f_train['Net_PnL'] < 0].sum()) > 0 else 0
        
        # Test PF
        sigs_f, probs_f = bt_falcon.generate_signals(X_test, meta_test)
        _, _, _, _, tr_f = get_fold_metrics(bt_falcon, X_test, meta_test, sigs_f, probs_f)
        test_pf = tr_f['Net_PnL'][tr_f['Net_PnL'] > 0].sum() / abs(tr_f['Net_PnL'][tr_f['Net_PnL'] < 0].sum()) if len(tr_f) > 0 and abs(tr_f['Net_PnL'][tr_f['Net_PnL'] < 0].sum()) > 0 else 0
        
        cumulative_trades['Falcon'].append(tr_f)
        
        # 2. Rule TWAP
        bt_twap = TradingBacktester(exp_skip_low_vol=True)
        bt_twap.threshold = 0.65
        probs_twap = np.where(meta_test['close'] > meta_test['twap'], 1.0, 0.0)
        sigs_twap = np.where(probs_twap > 0.65, 1, np.where(probs_twap < 0.35, -1, 0))
        _, _, _, _, tr_twap = get_fold_metrics(bt_twap, X_test, meta_test, sigs_twap, probs_twap)
        cumulative_trades['Rule_TWAP'].append(tr_twap)
        rule_pf = tr_twap['Net_PnL'][tr_twap['Net_PnL'] > 0].sum() / abs(tr_twap['Net_PnL'][tr_twap['Net_PnL'] < 0].sum()) if len(tr_twap) > 0 and abs(tr_twap['Net_PnL'][tr_twap['Net_PnL'] < 0].sum()) > 0 else 0
        
        # 3. Random Signals
        np.random.seed(42 + i)
        bt_rand = TradingBacktester(exp_skip_low_vol=True)
        bt_rand.threshold = 0.65
        probs_rand = np.random.uniform(0, 1, size=len(meta_test))
        sigs_rand = np.where(probs_rand > 0.65, 1, np.where(probs_rand < 0.35, -1, 0))
        _, _, _, _, tr_rand = get_fold_metrics(bt_rand, X_test, meta_test, sigs_rand, probs_rand)
        cumulative_trades['Random'].append(tr_rand)
        
        gen_gap = test_pf - train_pf
        win_icon = "Yes" if test_pf > rule_pf else "No"
        if test_pf > rule_pf: falcon_wins += 1
        
        print(f"| {total_folds} | {train_pf:.2f} | {test_pf:.2f} | {gen_gap:+.2f} | {rule_pf:.2f} | {win_icon} |")

    print(f"\n**Fold Dominance:** Falcon beat the Rule Baseline in {falcon_wins} out of {total_folds} folds.\n")

    print("## 3. Cumulative Out-Of-Sample Results\n")
    print("| Model Benchmark | Trades | Win Rate | Expectancy | PF | Sharpe | Return | Total PnL |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for name, tr_list in cumulative_trades.items():
        if len(tr_list) == 0: continue
        all_tr = pd.concat([t for t in tr_list if not t.empty], ignore_index=True) if any(not t.empty for t in tr_list) else pd.DataFrame()
        n, wr, pf, sh, ar, ret, exp = get_cumulative_metrics(all_tr)
        pnl = all_tr['Net_PnL'].sum() if not all_tr.empty else 0
        print(f"| **{name}** | {n} | {wr:.1f}% | {exp:.2f}R | {pf:.2f} | {sh:.2f} | {ret:.2f}% | Rs{pnl:,.0f} |")

if __name__ == "__main__":
    run_walk_forward()
