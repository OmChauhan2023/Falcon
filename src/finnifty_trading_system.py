"""
FINNIFTY ALGORITHMIC TRADING SYSTEM (v4.3 - Volatility Experiments)
Architecture: Regime-Switching XGBoost Pipeline
Features: Session Phase, TWAP Reversion, R-Multiple Trade Ledger, Indian Futures Sizing
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
import warnings
import glob
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Clean institutional theme (light) ────────────────────────────────────────
COLOR_BG        = 'white'
COLOR_GRID      = '#e6e6e6'
COLOR_TEXT      = '#212121'
COLOR_WIN       = '#2e7d32'
COLOR_LOSS      = '#c62828'
COLOR_DRAWDOWN  = '#c62828'
COLOR_SPOT      = '#0277bd'

plt.rcParams.update({
    'figure.facecolor':    COLOR_BG,
    'axes.facecolor':      COLOR_BG,
    'axes.grid':           True,
    'font.family':         'DejaVu Sans',
    'font.size':           10,
    'savefig.bbox':        'tight',
    'savefig.dpi':         160,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "raw", "Equity_1min")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# PART 1: DATA LOADING & SEGMENT IDENTIFICATION
# ==============================================================================

class DataLoader:
    def __init__(self, base_path=DATA_DIR):
        self.base_path = base_path

    def load_data(self, file_indices=None):
        all_files = sorted(glob.glob(os.path.join(self.base_path, "*.csv")))
        if file_indices is not None:
            all_files = [all_files[i] for i in file_indices if i < len(all_files)]

        dfs = []
        for file in tqdm(all_files, desc="Loading CSVs"):
            df = pd.read_csv(file)
            df['date'] = df['date'].astype(str).str.replace('="', '').str.replace('"', '')
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined['datetime'] = pd.to_datetime(
            combined['date'] + ' ' + combined['time'],
            format='%d-%m-%y %H:%M:%S', errors='coerce'
        )
        combined = combined.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

        gap_threshold = pd.Timedelta(days=5)
        gaps = combined['datetime'].diff()
        combined['segment'] = gaps.gt(gap_threshold).cumsum()
        
        combined['session'] = (combined['datetime'].dt.date != combined['datetime'].dt.date.shift()).cumsum()
        combined['hour']         = combined['datetime'].dt.hour
        combined['minute']       = combined['datetime'].dt.minute
        combined['time_decimal'] = combined['hour'] + combined['minute'] / 60

        combined = combined[
            (combined['time_decimal'] >= 9.25) &
            (combined['time_decimal'] <= 15.5)
        ].reset_index(drop=True)

        return combined

    def split_train_val_test(self):
        train = self.load_data(list(range(0, 12)))
        val   = self.load_data(list(range(12, 14)))
        test  = self.load_data(list(range(14, 17)))
        return train, val, test

# ==============================================================================
# PART 2: REGIME-AWARE FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    @staticmethod
    def _calc_segment_features(seg_df):
        df = seg_df.copy()
        tr = np.maximum(df['high'] - df['low'],
             np.maximum(abs(df['high'] - df['close'].shift(1)),
                        abs(df['low']  - df['close'].shift(1))))
        df['atr'] = tr.rolling(20).mean()
        df['returns'] = df['close'].pct_change()
        df['rv_30m']  = df['returns'].rolling(30).std() * np.sqrt(375)
        df['rv_1d']   = df['returns'].rolling(375).std() * np.sqrt(375*252)
        df['atr_normalized_return'] = (df['close'] - df['close'].shift(1)) / (df['atr'] + 1e-10)
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = mid + 2 * std
        df['bb_lower'] = mid - 2 * std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (mid + 1e-10)
        df['bb_squeeze_pct'] = df['bb_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=True
        )
        df['sma_200'] = df['close'].rolling(200 * 375).mean()
        df['dist_from_200dma'] = (df['close'] - df['sma_200']) / (df['atr'] * np.sqrt(200) + 1e-10)
        df['close_pos_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        min_date = df['datetime'].min()
        df['days_since_gap'] = (df['datetime'] - min_date).dt.days
        
        # Categorical Volatility Features for Experiment 3
        df['vol_regime_LOW'] = (df['bb_squeeze_pct'] < 0.30).astype(int)
        df['vol_regime_MID'] = ((df['bb_squeeze_pct'] >= 0.30) & (df['bb_squeeze_pct'] <= 0.70)).astype(int)
        df['vol_regime_HIGH'] = (df['bb_squeeze_pct'] > 0.70).astype(int)
        
        return df

    @staticmethod
    def _calc_session_features(sess_df):
        df = sess_df.copy()
        df['twap'] = df['close'].cumsum() / (np.arange(len(df)) + 1)
        df['dist_from_twap'] = (df['close'] - df['twap']) / (df['atr'].shift(1).fillna(10.0))
        if len(df) >= 15:
            orb_high = df.iloc[:15]['high'].max()
            orb_low  = df.iloc[:15]['low'].min()
        else:
            orb_high = df['high'].max()
            orb_low  = df['low'].min()
        df['dist_from_orb_high'] = (df['close'] - orb_high) / (df['atr'] + 1e-10)
        df['dist_from_orb_low']  = (df['close'] - orb_low) / (df['atr'] + 1e-10)
        return df

    def engineer_all_features(self, df, target_horizon=5):
        df = df.copy()
        df['phase_open']   = (df['time_decimal'] < 10.25).astype(int)
        df['phase_midday'] = ((df['time_decimal'] >= 10.25) & (df['time_decimal'] < 13.5)).astype(int)
        df['phase_close']  = (df['time_decimal'] >= 13.5).astype(int)
        df['prev_close'] = df['close'].shift(1)
        df['is_new_session'] = (df['session'] != df['session'].shift(1))
        df['gap_size'] = np.where(df['is_new_session'], (df['open'] - df['prev_close']) / (df['prev_close'] + 1e-10), 0.0)
        df['gap_size'] = df.groupby('session')['gap_size'].transform('first')
        df = df.groupby('segment', group_keys=False).apply(self._calc_segment_features).reset_index(drop=True)
        df = df.groupby('session', group_keys=False).apply(self._calc_session_features).reset_index(drop=True)
        df['future_return']    = df['close'].shift(-target_horizon) / df['close'] - 1
        df['future_direction'] = (df['future_return'] > 0).astype(int)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['atr', 'rv_30m', 'dist_from_twap'])
        return df.reset_index(drop=True)

# ==============================================================================
# PART 3: DATASET PREPARATION
# ==============================================================================

class DatasetPreparer:
    @staticmethod
    def get_feature_columns(include_categorical_vol=False):
        cols = [
            'phase_open', 'phase_midday', 'phase_close',
            'gap_size', 'rv_30m', 'rv_1d', 'atr_normalized_return',
            'bb_width', 'bb_squeeze_pct', 'dist_from_200dma',
            'close_pos_in_range', 'days_since_gap',
            'dist_from_twap', 'dist_from_orb_high', 'dist_from_orb_low'
        ]
        if include_categorical_vol:
            cols += ['vol_regime_LOW', 'vol_regime_MID', 'vol_regime_HIGH']
        return cols

    @staticmethod
    def prepare_ml_dataset(df, feature_cols=None):
        if feature_cols is None:
            feature_cols = DatasetPreparer.get_feature_columns(include_categorical_vol=False)
        target_col   = 'future_direction'
        metadata_cols = ['datetime','open','high','low','close','volume',
                         'future_return', 'atr', 'segment', 'session',
                         'twap', 'dist_from_twap', 'phase_midday', 'phase_close',
                         'bb_squeeze_pct', 'dist_from_orb_high', 'dist_from_orb_low']

        col_list = list(dict.fromkeys(feature_cols + [target_col] + metadata_cols))
        df_sub   = df[[c for c in col_list if c in df.columns]].copy()
        
        df_clean = df_sub.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
        X        = df_clean[feature_cols]
        y        = df_clean[target_col]
        meta     = df_clean[[c for c in metadata_cols if c in df_clean.columns]]
        return X, y, meta, feature_cols

# ==============================================================================
# PART 4: MODEL TRAINING
# ==============================================================================

class XGBoostTrainer:
    def __init__(self):
        self.model           = None
        self.scaler          = None
        self.best_threshold  = 0.65
        self.short_threshold = 0.35

    def scale_features(self, X_train, X_val, X_test):
        self.scaler = StandardScaler()
        return (self.scaler.fit_transform(X_train),
                self.scaler.transform(X_val),
                self.scaler.transform(X_test))

    def train_model(self, X_train, y_train, X_val, y_val, verbose=False):
        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos

        params = {
            'objective':            'binary:logistic',
            'eval_metric':          ['logloss', 'auc'],
            'max_depth':            5,
            'learning_rate':        0.02,
            'n_estimators':         1000,
            'min_child_weight':     10,
            'subsample':            0.7,
            'colsample_bytree':     0.7,
            'gamma':                0.5,
            'reg_alpha':            3.0,
            'reg_lambda':           5.0,
            'scale_pos_weight':     scale_pos_weight,
            'random_state':         42,
            'n_jobs':               -1,
            'tree_method':          'hist',
            'early_stopping_rounds': 30,
        }
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
        return self.model

# ==============================================================================
# PART 5: THE INDIAN MARKET EXECUTION ENGINE
# ==============================================================================

class TradingBacktester:
    def __init__(self, initial_capital=1000000, risk_per_trade=0.005,
                 lot_size=40, brokerage_per_order=20, statutory_tax_pct=0.0003,
                 slippage_points=1.0, cooldown_bars=10, time_stop_bars=30,
                 max_margin_pct=0.80, 
                 exp_skip_low_vol=False, exp_twap_exit=False):
        
        self.initial_capital  = initial_capital
        self.risk_per_trade   = risk_per_trade
        self.lot_size         = lot_size
        self.brokerage_per_order = brokerage_per_order
        self.statutory_tax_pct   = statutory_tax_pct
        self.slippage_points     = slippage_points
        self.cooldown_bars    = cooldown_bars
        self.time_stop_bars   = time_stop_bars
        self.max_margin_pct   = max_margin_pct
        
        # Experiments
        self.exp_skip_low_vol = exp_skip_low_vol
        self.exp_twap_exit    = exp_twap_exit
        
        self.model = self.scaler = None
        self.threshold       = 0.65    
        self.short_threshold = 0.35    

    def generate_signals(self, X, meta):
        X_scaled      = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        prob_long = probabilities > self.threshold
        prob_short = probabilities < self.short_threshold

        dist_twap = meta['dist_from_twap'].values
        is_midday = meta['phase_midday'].values == 1
        is_close  = meta['phase_close'].values == 1

        midday_long  = is_midday & prob_long & (dist_twap < 0)
        midday_short = is_midday & prob_short & (dist_twap > 0)
        close_long  = is_close & prob_long & (dist_twap > 0)
        close_short = is_close & prob_short & (dist_twap < 0)

        signals = np.zeros(len(meta), dtype=int)
        signals[midday_long | close_long] = 1
        signals[midday_short | close_short] = -1

        return signals, probabilities

    def backtest(self, X, meta, signals, probabilities):
        capital      = self.initial_capital
        equity_curve = [capital]
        trades       = []

        in_position       = False
        direction         = 0             
        entry_price       = 0.0
        position_size     = 0
        stop_loss         = 0.0
        target_price      = 0.0
        risk_per_share    = 0.0
        entry_idx         = 0
        entry_prob        = 0.0
        entry_regime      = ""
        entry_session     = ""
        extreme_since     = 0.0           
        last_exit_bar     = -10**9

        tax_per_leg = self.statutory_tax_pct / 2.0

        def realize_exit(exit_idx, exit_price_raw, exit_reason):
            nonlocal capital, in_position, last_exit_bar, position_size, direction
            ex_price = exit_price_raw - (self.slippage_points * direction)
            entry_turnover = entry_price * position_size
            exit_turnover  = ex_price * position_size
            entry_cost = self.brokerage_per_order + (entry_turnover * tax_per_leg)
            exit_cost  = self.brokerage_per_order + (exit_turnover * tax_per_leg)
            total_costs = entry_cost + exit_cost
            
            gross_pnl = (ex_price - entry_price) * position_size * direction
            net_pnl = gross_pnl - total_costs
            capital += net_pnl
            
            realized_r = gross_pnl / (risk_per_share * position_size) if risk_per_share > 0 else 0

            trades.append({
                'Date':           meta.iloc[entry_idx]['datetime'].date(),
                'Entry_Time':     meta.iloc[entry_idx]['datetime'].time(),
                'Exit_Time':      meta.iloc[exit_idx]['datetime'].time(),
                'Regime':         entry_regime,
                'Session':        entry_session,
                'Direction':      'LONG' if direction == 1 else 'SHORT',
                'Probability':    round(entry_prob, 3),
                'Lots':           int(position_size / self.lot_size),
                'Entry_Price':    round(entry_price, 2),
                'Exit_Price':     round(ex_price, 2),
                'Net_PnL':        round(net_pnl, 2),
                'R':              round(realized_r, 2),
                'Exit_Reason':    exit_reason
            })
            in_position = False; last_exit_bar = exit_idx; position_size = 0; direction = 0

        for i in range(len(meta)):
            row   = meta.iloc[i]
            price = row['close']
            high  = row['high']
            low   = row['low']

            if in_position:
                atr_now = row['atr'] if pd.notna(row['atr']) else (price * 0.001)
                vol_pct_now = row['bb_squeeze_pct'] if pd.notna(row['bb_squeeze_pct']) else 0.5
                
                # Dynamic Trailing
                sl_trail_mult = 1.0 if vol_pct_now < 0.30 else (1.5 if vol_pct_now > 0.70 else 1.0)
                if direction == 1:
                    extreme_since = max(extreme_since, high)
                    stop_loss = max(stop_loss, extreme_since - sl_trail_mult * atr_now)
                else: 
                    extreme_since = min(extreme_since, low)
                    stop_loss = min(stop_loss, extreme_since + sl_trail_mult * atr_now)

                bars_held = i - entry_idx
                
                hit_target = False
                # Experiment 2: TWAP Exit for Low Vol
                if self.exp_twap_exit and entry_regime == "Low Vol":
                    twap_now = row['twap']
                    if direction == 1 and high >= twap_now and twap_now > entry_price:
                        target_price = twap_now
                        hit_target = True
                    elif direction == -1 and low <= twap_now and twap_now < entry_price:
                        target_price = twap_now
                        hit_target = True
                else:
                    hit_target = (high >= target_price) if direction == 1 else (low <= target_price)
                
                stop_hit = (low <= stop_loss) if direction == 1 else (high >= stop_loss)

                if hit_target:
                    realize_exit(i, target_price, "Target Reached")
                elif stop_hit:
                    realize_exit(i, stop_loss, "Stop Loss")
                elif bars_held >= self.time_stop_bars:
                    realize_exit(i, price, "Time Stop")
                elif i == len(meta) - 1:
                    realize_exit(i, price, "End of Data")

            entry_candidate = (not in_position) and (signals[i] != 0) and ((i - last_exit_bar) > self.cooldown_bars)
            
            if entry_candidate:
                sig = signals[i]
                vol_pct = row['bb_squeeze_pct'] if pd.notna(row['bb_squeeze_pct']) else 0.5
                
                # Experiment 1: Filter Low Vol completely
                if self.exp_skip_low_vol and vol_pct < 0.30:
                    continue
                
                if vol_pct < 0.30:
                    tp_atr = 1.0; sl_atr = 1.0
                    rgm = "Low Vol"
                elif vol_pct > 0.70:
                    tp_atr = 4.0; sl_atr = 1.5
                    rgm = "High Vol"
                else:
                    tp_atr = 2.0; sl_atr = 1.0
                    rgm = "Mid Vol"

                entry_price_raw = price + (self.slippage_points * sig)
                atr_val = row['atr'] if pd.notna(row['atr']) else (price * 0.001)
                r_pts = sl_atr * atr_val
                
                if r_pts > 0:
                    confidence = abs(probabilities[i] - 0.5)
                    size_mult = 1 + 3 * confidence
                    if row['phase_close'] == 1:
                        if sig == 1 and row['dist_from_orb_high'] > 0: size_mult *= 1.5
                        if sig == -1 and row['dist_from_orb_low'] < 0: size_mult *= 1.5

                    max_risk_inr = capital * self.risk_per_trade * size_mult
                    lots = np.floor(max_risk_inr / (r_pts * self.lot_size))
                    
                    margin_per_lot = entry_price_raw * self.lot_size * 0.12
                    max_margin_lots = np.floor((capital * self.max_margin_pct) / margin_per_lot)
                    
                    lots = min(lots, max_margin_lots)
                    
                    if lots >= 1:
                        position_size    = lots * self.lot_size
                        entry_price      = entry_price_raw
                        risk_per_share   = r_pts
                        stop_loss        = entry_price - (sig * r_pts)
                        target_price     = entry_price + (sig * tp_atr * atr_val)
                        
                        direction        = sig
                        extreme_since    = high if sig == 1 else low
                        in_position      = True
                        entry_idx        = i
                        entry_prob       = probabilities[i]
                        entry_regime     = rgm
                        entry_session    = "Close" if row['phase_close'] == 1 else "Midday"
                        
            if in_position:
                equity_curve.append(capital + (price - entry_price) * position_size * direction)
            else:
                equity_curve.append(capital)

        trades_df = pd.DataFrame(trades)
        return trades_df, np.array(equity_curve)

# ==============================================================================
# PART 6: METRICS
# ==============================================================================

class PerformanceEvaluator:
    @staticmethod
    def generate_r_report(trades_df, equity_curve, initial_capital, label="Report"):
        print(f"\n[{label.upper()}]")
        if len(trades_df) == 0:
            print("  No trades executed."); return {}

        returns_pct = pd.Series(equity_curve).pct_change().dropna()
        sharpe  = returns_pct.mean() / returns_pct.std() * np.sqrt(252 * 375) if returns_pct.std() > 0 else 0
        eq      = np.array(equity_curve)
        peak    = np.maximum.accumulate(eq)
        max_dd  = ((eq - peak) / peak).min()
        total_r = (eq[-1] - initial_capital) / initial_capital * 100

        wins       = trades_df[trades_df['Net_PnL'] > 0]
        losses     = trades_df[trades_df['Net_PnL'] < 0]
        win_rate   = len(wins) / len(trades_df) * 100 if len(trades_df) else 0
        gross_p    = wins['Net_PnL'].sum()
        gross_l    = abs(losses['Net_PnL'].sum())
        pf         = gross_p / gross_l if gross_l > 0 else float('inf')
        
        avg_r      = trades_df['R'].mean() if len(trades_df) else 0
        print(f"  Trades: {len(trades_df):<4} | Win Rate: {win_rate:>5.1f}% | PF: {pf:>5.2f} | Sharpe: {sharpe:>5.2f} | Avg R: {avg_r:>5.2f}R | Return: {total_r:>6.2f}%")
        
        return {'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate, 'pf': pf, 'avg_r': avg_r, 'return': total_r}

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  VOLATILITY REGIME EXPERIMENTS")
    print("="*70)

    loader   = DataLoader()
    engineer = FeatureEngineer()
    preparer = DatasetPreparer()

    train_raw, val_raw, test_raw = loader.split_train_val_test()

    train_df = engineer.engineer_all_features(train_raw)
    val_df   = engineer.engineer_all_features(val_raw)
    test_df  = engineer.engineer_all_features(test_raw)

    print("\n--- BASELINE MODEL TRAINING ---")
    X_train, y_train, train_meta, feat_cols = preparer.prepare_ml_dataset(train_df, DatasetPreparer.get_feature_columns(False))
    X_val,   y_val,   val_meta,   _         = preparer.prepare_ml_dataset(val_df,   feat_cols)
    X_test,  y_test,  test_meta,  _         = preparer.prepare_ml_dataset(test_df,  feat_cols)

    if len(X_val) == 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        val_meta = train_meta.iloc[X_val.index].copy()
    if len(X_test) == 0:
        X_test, y_test, test_meta = X_val.copy(), y_val.copy(), val_meta.copy()

    trainer_base = XGBoostTrainer()
    X_tr_s, X_v_s, X_te_s = trainer_base.scale_features(X_train, X_val, X_test)
    model_base = trainer_base.train_model(X_tr_s, y_train, X_v_s, y_val)
    
    print("\n--- EXPERIMENT 3 MODEL TRAINING (CATEGORICAL) ---")
    feat_cols_cat = DatasetPreparer.get_feature_columns(include_categorical_vol=True)
    X_train_cat, y_train_cat, _, _ = preparer.prepare_ml_dataset(train_df, feat_cols_cat)
    X_val_cat,   y_val_cat,   _, _ = preparer.prepare_ml_dataset(val_df,   feat_cols_cat)
    X_test_cat,  y_test_cat,  _, _ = preparer.prepare_ml_dataset(test_df,  feat_cols_cat)
    
    if len(X_val_cat) == 0:
        X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
            X_train_cat, y_train_cat, test_size=0.2, random_state=42, stratify=y_train_cat)
    if len(X_test_cat) == 0:
        X_test_cat = X_val_cat.copy()
        
    trainer_cat = XGBoostTrainer()
    X_tr_s_c, X_v_s_c, X_te_s_c = trainer_cat.scale_features(X_train_cat, X_val_cat, X_test_cat)
    model_cat = trainer_cat.train_model(X_tr_s_c, y_train_cat, X_v_s_c, y_val_cat)
    
    print("\n======================================================================")
    print("  EXPERIMENT RESULTS (Test Set)")
    print("======================================================================")
    
    evaluator = PerformanceEvaluator()
    
    # BASELINE
    bt_base = TradingBacktester()
    bt_base.model, bt_base.scaler = model_base, trainer_base.scaler
    sigs, probs = bt_base.generate_signals(X_test, test_meta)
    tr_base, eq_base = bt_base.backtest(X_test, test_meta, sigs, probs)
    evaluator.generate_r_report(tr_base, eq_base, bt_base.initial_capital, label="Baseline (No Filters)")
    
    # EXP 1: FILTER LOW VOL
    bt_exp1 = TradingBacktester(exp_skip_low_vol=True)
    bt_exp1.model, bt_exp1.scaler = model_base, trainer_base.scaler
    sigs1, probs1 = bt_exp1.generate_signals(X_test, test_meta)
    tr_exp1, eq_exp1 = bt_exp1.backtest(X_test, test_meta, sigs1, probs1)
    evaluator.generate_r_report(tr_exp1, eq_exp1, bt_exp1.initial_capital, label="Exp 1: Filter Low Vol")
    
    # EXP 2: TWAP EXIT FOR LOW VOL
    bt_exp2 = TradingBacktester(exp_twap_exit=True)
    bt_exp2.model, bt_exp2.scaler = model_base, trainer_base.scaler
    sigs2, probs2 = bt_exp2.generate_signals(X_test, test_meta)
    tr_exp2, eq_exp2 = bt_exp2.backtest(X_test, test_meta, sigs2, probs2)
    evaluator.generate_r_report(tr_exp2, eq_exp2, bt_exp2.initial_capital, label="Exp 2: Mean Reversion Exits")

    # EXP 3: CATEGORICAL MODEL
    bt_exp3 = TradingBacktester()
    bt_exp3.model, bt_exp3.scaler = model_cat, trainer_cat.scaler
    sigs3, probs3 = bt_exp3.generate_signals(X_test_cat, test_meta)
    tr_exp3, eq_exp3 = bt_exp3.backtest(X_test_cat, test_meta, sigs3, probs3)
    evaluator.generate_r_report(tr_exp3, eq_exp3, bt_exp3.initial_capital, label="Exp 3: Categorical Model Retraining")

    print("\n" + "="*70)
    print("  EXPERIMENTS COMPLETE")
    print("="*70)
