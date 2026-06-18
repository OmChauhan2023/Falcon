import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from finnifty_trading_system import DataLoader, FeatureEngineer, DatasetPreparer, XGBoostTrainer, TradingBacktester

REPORT_FILE = os.path.join(BASE_DIR, "outputs", "reports", "validation_gauntlet_report.md")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
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

def get_metrics(bt, X, meta, sigs, probs):
    trades_df, eq_curve = bt.backtest(X, meta, sigs, probs)
    if len(trades_df) == 0:
        return 0, 0, 0, 0, 0, 0, trades_df
        
    returns_pct = pd.Series(eq_curve).pct_change().dropna()
    sharpe = returns_pct.mean() / returns_pct.std() * np.sqrt(252 * 375) if returns_pct.std() > 0 else 0
    total_r = (eq_curve[-1] - bt.initial_capital) / bt.initial_capital * 100
    
    wins = trades_df[trades_df['Net_PnL'] > 0]
    losses = trades_df[trades_df['Net_PnL'] < 0]
    win_rate = len(wins) / len(trades_df) * 100
    pf = wins['Net_PnL'].sum() / abs(losses['Net_PnL'].sum()) if abs(losses['Net_PnL'].sum()) > 0 else float('inf')
    avg_r = trades_df['R'].mean()
    expectancy = calc_expectancy(trades_df)
    
    return len(trades_df), win_rate, pf, sharpe, avg_r, total_r, expectancy, trades_df

def run_gauntlet():
    sys.stdout = Logger(REPORT_FILE)
    
    print("# TEAM FALCON: VALIDATION GAUNTLET REPORT\n")
    print("> Goal: Relentlessly attack the trading system to uncover hidden curve-fitting or structural fragilities.\n")

    print("## Initialization: Training Baseline Engine")
    loader = DataLoader()
    train_raw, val_raw, test_raw = loader.split_train_val_test()

    engineer = FeatureEngineer()
    train_df = engineer.engineer_all_features(train_raw)
    val_df   = engineer.engineer_all_features(val_raw)
    test_df  = engineer.engineer_all_features(test_raw)

    preparer = DatasetPreparer()
    feat_cols = DatasetPreparer.get_feature_columns(False)
    X_train, y_train, train_meta, _ = preparer.prepare_ml_dataset(train_df, feat_cols)
    X_val,   y_val,   val_meta,   _ = preparer.prepare_ml_dataset(val_df, feat_cols)
    X_test,  y_test,  test_meta,  _ = preparer.prepare_ml_dataset(test_df, feat_cols)

    if len(X_val) == 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        val_meta = train_meta.iloc[X_val.index].copy()
    if len(X_test) == 0:
        X_test, y_test, test_meta = X_val.copy(), y_val.copy(), val_meta.copy()

    trainer = XGBoostTrainer()
    X_tr_s, X_v_s, X_te_s = trainer.scale_features(X_train, X_val, X_test)
    model = trainer.train_model(X_tr_s, y_train, X_v_s, y_val, verbose=False)
    print("Baseline model trained.\n")
    
    # Precompute signals and probabilities
    bt_base = TradingBacktester(exp_skip_low_vol=True)
    bt_base.model, bt_base.scaler = model, trainer.scaler
    sigs_base, probs_base = bt_base.generate_signals(X_test, test_meta)

    # Base Metrics
    n, wr, pf, sh, ar, ret, exp, base_trades = get_metrics(bt_base, X_test, test_meta, sigs_base, probs_base)
    print("### Global Test Set (Baseline Filtered)")
    print(f"- **Trades:** {n} | **Win Rate:** {wr:.1f}% | **PF:** {pf:.2f} | **Sharpe:** {sh:.2f}")
    print(f"- **Expectancy per Trade:** {exp:.2f}R | **Return:** {ret:.2f}%\n")
    
    print("---\n")
    
    # 1. Segment Decomposition
    print("## 1. Segment Decomposition")
    print("Evaluating stability across the 3 independent test segments.\n")
    
    segments = test_meta['segment'].unique()
    for seg in segments:
        mask = test_meta['segment'] == seg
        if mask.sum() == 0: continue
        X_sub = X_test[mask]
        meta_sub = test_meta[mask]
        sigs_sub = sigs_base[mask]
        probs_sub = probs_base[mask]
        
        n_s, wr_s, pf_s, sh_s, ar_s, ret_s, exp_s, _ = get_metrics(bt_base, X_sub, meta_sub, sigs_sub, probs_sub)
        print(f"- **Segment {seg}:** Trades: {n_s:<3} | PF: {pf_s:>5.2f} | Sharpe: {sh_s:>5.2f} | Expectancy: {exp_s:>5.2f}R | Return: {ret_s:>5.2f}%")
        
    print("\n---\n")

    # 2. Temporal Stability
    print("## 2. Temporal Stability")
    print("Evaluating chronological decay across the test set (First 33%, Middle 33%, Last 33%).\n")
    
    thirds = np.array_split(np.arange(len(test_meta)), 3)
    labels = ["First 33%", "Middle 33%", "Last 33%"]
    
    for label, idxs in zip(labels, thirds):
        if len(idxs) == 0: continue
        X_sub = X_test.iloc[idxs]
        meta_sub = test_meta.iloc[idxs]
        sigs_sub = sigs_base[idxs]
        probs_sub = probs_base[idxs]
        
        n_s, wr_s, pf_s, sh_s, ar_s, ret_s, exp_s, _ = get_metrics(bt_base, X_sub, meta_sub, sigs_sub, probs_sub)
        print(f"- **{label}:** Trades: {n_s:<3} | PF: {pf_s:>5.2f} | Sharpe: {sh_s:>5.2f} | Expectancy: {exp_s:>5.2f}R")

    print("\n---\n")

    # 3. Threshold Stability
    print("## 3. Threshold Stability")
    print("Testing classification boundaries to detect curve-fitting.\n")
    
    thresholds = [0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.72]
    print("| Threshold | Trades | Win Rate | Expectancy | PF | Sharpe | Return |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for t in thresholds:
        bt_t = TradingBacktester(exp_skip_low_vol=True)
        bt_t.model, bt_t.scaler = model, trainer.scaler
        bt_t.threshold = t
        bt_t.short_threshold = 1 - t
        
        s_t, p_t = bt_t.generate_signals(X_test, test_meta)
        n_t, wr_t, pf_t, sh_t, ar_t, ret_t, exp_t, _ = get_metrics(bt_t, X_test, test_meta, s_t, p_t)
        
        print(f"| {t:.2f} | {n_t} | {wr_t:.1f}% | {exp_t:.2f}R | {pf_t:.2f} | {sh_t:.2f} | {ret_t:.2f}% |")

    print("\n---\n")

    # 4. Slippage Stress Matrix
    print("## 4. Slippage Stress Matrix")
    print("Evaluating survival against extreme liquidity friction.\n")
    
    slippages = [1.0, 2.0, 3.0, 4.0, 5.0]
    print("| Slippage (pts) | Trades | Win Rate | Expectancy | PF | Sharpe | Return |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for slip in slippages:
        bt_s = TradingBacktester(exp_skip_low_vol=True, slippage_points=slip)
        bt_s.model, bt_s.scaler = model, trainer.scaler
        
        n_s, wr_s, pf_s, sh_s, ar_s, ret_s, exp_s, _ = get_metrics(bt_s, X_test, test_meta, sigs_base, probs_base)
        print(f"| {slip:.1f} | {n_s} | {wr_s:.1f}% | {exp_s:.2f}R | {pf_s:.2f} | {sh_s:.2f} | {ret_s:.2f}% |")

    print("\n---\n")

    # 5. Trade Concentration
    print("## 5. Trade Concentration")
    print("Checking dependency on outlier lottery-ticket trades.\n")
    
    if len(base_trades) > 0:
        base_trades_sorted = base_trades.sort_values('Net_PnL', ascending=False).reset_index(drop=True)
        total_pnl = base_trades['Net_PnL'].sum()
        
        def exclude_top(pct):
            exclude_count = int(np.ceil(len(base_trades) * pct))
            excluded = base_trades_sorted.iloc[:exclude_count]
            remaining = base_trades_sorted.iloc[exclude_count:]
            exc_pnl = excluded['Net_PnL'].sum()
            rem_pnl = remaining['Net_PnL'].sum()
            return exc_pnl, rem_pnl
            
        print("| Excluded Outliers | Removed PnL | Remaining PnL | Retained % | Profitable? |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        
        for p in [0.01, 0.05, 0.10]:
            exc, rem = exclude_top(p)
            ret_pct = (rem / total_pnl) * 100 if total_pnl > 0 else 0
            is_prof = "Yes" if rem > 0 else "No"
            print(f"| Top {int(p*100)}% | Rs{exc:,.0f} | Rs{rem:,.0f} | {ret_pct:.1f}% | {is_prof} |")

if __name__ == "__main__":
    run_gauntlet()
