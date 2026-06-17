"""
FINNIFTY ALGORITHMIC TRADING SYSTEM
Structure-based Trading: OTE (Fibonacci 0.5-0.7) + FVG + Market Regime
ML Model: XGBoost | Exit: Chandelier Stop + 2:1 Take Profit
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — never opens a GUI window
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
import warnings
import glob
import pickle
from datetime import timedelta
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

warnings.filterwarnings('ignore')

# ── Clean institutional theme (light) ────────────────────────────────────────
COLOR_BG        = 'white'
COLOR_GRID      = '#e6e6e6'
COLOR_TEXT      = '#212121'
COLOR_DIM       = '#9e9e9e'
COLOR_NEUTRAL   = '#546e7a'
COLOR_WIN       = '#2e7d32'   # emerald
COLOR_LOSS      = '#c62828'   # crimson
COLOR_DRAWDOWN  = '#c62828'
COLOR_EQUITY    = '#1f3a6f'   # navy
COLOR_ACCENT    = '#ef6c00'   # amber
COLOR_PURPLE    = '#6a1b9a'
COLOR_MAGENTA   = '#ad1457'
COLOR_SPOT      = '#0277bd'   # blue for FINNIFTY price line

plt.rcParams.update({
    'figure.facecolor':    COLOR_BG,
    'axes.facecolor':      COLOR_BG,
    'axes.edgecolor':      '#cfcfcf',
    'axes.labelcolor':     COLOR_TEXT,
    'axes.titlecolor':     COLOR_TEXT,
    'axes.titleweight':    'bold',
    'axes.titlesize':      12,
    'axes.labelsize':      10,
    'axes.linewidth':      0.8,
    'axes.grid':           True,
    'axes.axisbelow':      True,
    'grid.color':          COLOR_GRID,
    'grid.linewidth':      0.7,
    'grid.alpha':          0.85,
    'xtick.color':         COLOR_TEXT,
    'ytick.color':         COLOR_TEXT,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
    'text.color':          COLOR_TEXT,
    'font.family':         'DejaVu Sans',
    'font.size':           10,
    'legend.frameon':      False,
    'legend.fontsize':     9,
    'savefig.facecolor':   COLOR_BG,
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
# PART 1: DATA LOADING
# ==============================================================================

class DataLoader:
    def __init__(self, base_path=DATA_DIR):
        self.base_path = base_path

    def load_data(self, file_indices=None):
        all_files = sorted(glob.glob(os.path.join(self.base_path, "*.csv")))

        if file_indices is not None:
            all_files = [all_files[i] for i in file_indices if i < len(all_files)]

        print(f"Loading {len(all_files)} file(s)...")

        dfs = []
        for file in tqdm(all_files):
            df = pd.read_csv(file)
            df['date'] = df['date'].astype(str).str.replace('="', '').str.replace('"', '')
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined['datetime'] = pd.to_datetime(
            combined['date'] + ' ' + combined['time'],
            format='%d-%m-%y %H:%M:%S', errors='coerce'
        )
        combined = combined.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

        combined['hour']         = combined['datetime'].dt.hour
        combined['minute']       = combined['datetime'].dt.minute
        combined['time_decimal'] = combined['hour'] + combined['minute'] / 60

        combined = combined[
            (combined['time_decimal'] >= 9.25) &
            (combined['time_decimal'] <= 15.5)
        ].reset_index(drop=True)

        print(f"Loaded {len(combined):,} records | {combined['datetime'].min()} to {combined['datetime'].max()}")
        return combined

    def split_train_val_test(self):
        print("\n" + "="*70)
        print("SPLITTING DATA: TRAIN / VAL / TEST")
        print("="*70)
        train = self.load_data(list(range(0, 12)))
        val   = self.load_data(list(range(12, 14)))
        test  = self.load_data(list(range(14, 17)))
        print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        return train, val, test

# ==============================================================================
# PART 2: FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:

    @staticmethod
    def calculate_fibonacci_levels(df, lookback=100):
        df = df.copy()
        df['swing_high_price'] = df['high'].rolling(lookback).max()
        df['swing_low_price']  = df['low'].rolling(lookback).min()
        df['fib_range']  = df['swing_high_price'] - df['swing_low_price']
        df['fib_0.236']  = df['swing_high_price'] - 0.236 * df['fib_range']
        df['fib_0.382']  = df['swing_high_price'] - 0.382 * df['fib_range']
        df['fib_0.500']  = df['swing_high_price'] - 0.500 * df['fib_range']
        df['fib_0.618']  = df['swing_high_price'] - 0.618 * df['fib_range']
        df['fib_0.700']  = df['swing_high_price'] - 0.700 * df['fib_range']
        df['fib_0.786']  = df['swing_high_price'] - 0.786 * df['fib_range']
        df['fib_0.600']  = df['swing_high_price'] - 0.600 * df['fib_range']
        tolerance = df['fib_range'] * 0.01
        df['in_ote_zone'] = (
            (df['close'] >= (df['fib_0.700'] - tolerance)) &
            (df['close'] <= (df['fib_0.500'] + tolerance))
        )
        df['distance_from_ote']  = (df['close'] - df['fib_0.600']) / df['fib_range']
        df['in_premium_zone']    = (df['close'] > df['fib_0.500']).astype(int)
        df['in_discount_zone']   = (df['close'] < df['fib_0.500']).astype(int)
        return df

    @staticmethod
    def identify_fvg(df, min_gap_pct=0.05):
        df = df.copy()
        df['bullish_fvg'] = (df['low'].shift(-2) - df['high']) / df['close'] > min_gap_pct / 100
        df['bearish_fvg'] = (df['low'] - df['high'].shift(-2)) / df['close'] > min_gap_pct / 100
        df['fvg_high'] = np.where(df['bullish_fvg'], df['low'].shift(-2),
                         np.where(df['bearish_fvg'], df['low'], np.nan))
        df['fvg_low']  = np.where(df['bullish_fvg'], df['high'],
                         np.where(df['bearish_fvg'], df['high'].shift(-2), np.nan))
        df['fvg_high_ffill'] = df['fvg_high'].ffill(limit=20)
        df['fvg_low_ffill']  = df['fvg_low'].ffill(limit=20)
        df['in_fvg']  = (df['close'] >= df['fvg_low_ffill']) & (df['close'] <= df['fvg_high_ffill'])
        df['fvg_size'] = ((df['fvg_high_ffill'] - df['fvg_low_ffill']) / df['close']).fillna(0)
        return df

    @staticmethod
    def calculate_market_regime(df, adx_period=14, trend_threshold=20):
        df = df.copy()
        df['tr'] = np.maximum(df['high'] - df['low'],
                   np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low']  - df['close'].shift(1))))
        df['high_diff'] = df['high'] - df['high'].shift(1)
        df['low_diff']  = df['low'].shift(1) - df['low']
        df['plus_dm']   = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
        df['minus_dm']  = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),  df['low_diff'],  0)
        df['atr']       = df['tr'].ewm(alpha=1/adx_period, adjust=False).mean()
        df['plus_di']   = 100 * (df['plus_dm'].ewm(alpha=1/adx_period, adjust=False).mean()  / df['atr'])
        df['minus_di']  = 100 * (df['minus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / df['atr'])
        df['dx']        = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx']       = df['dx'].ewm(alpha=1/adx_period, adjust=False).mean()
        df['sma_50']    = df['close'].rolling(50).mean()
        df['sma_200']   = df['close'].rolling(200).mean()
        df['market_regime'] = 'RANGING'
        trending = df['adx'] > trend_threshold
        df.loc[trending, 'market_regime'] = 'TRENDING'
        df.loc[trending & (df['plus_di'] > df['minus_di']) & (df['close'] > df['sma_50']), 'market_regime'] = 'STRONG_UPTREND'
        df.loc[trending & (df['minus_di'] > df['plus_di']) & (df['close'] < df['sma_50']), 'market_regime'] = 'STRONG_DOWNTREND'
        df['regime_ranging']    = (df['market_regime'] == 'RANGING').astype(int)
        df['regime_trending']   = (df['market_regime'] == 'TRENDING').astype(int)
        df['regime_strong_up']  = (df['market_regime'] == 'STRONG_UPTREND').astype(int)
        df['regime_strong_down']= (df['market_regime'] == 'STRONG_DOWNTREND').astype(int)
        return df

    @staticmethod
    def detect_overstretched_move(df, lookback=14, threshold=2):
        df = df.copy()
        delta = df['close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(lookback).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df['bb_middle'] = df['close'].rolling(lookback).mean()
        df['bb_std']    = df['close'].rolling(lookback).std()
        df['bb_upper']  = df['bb_middle'] + threshold * df['bb_std']
        df['bb_lower']  = df['bb_middle'] - threshold * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        if 'atr' not in df.columns:
            tr = np.maximum(df['high'] - df['low'],
                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                            abs(df['low']  - df['close'].shift(1))))
            df['atr'] = tr.rolling(lookback).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['price_distance_atr'] = abs(df['close'] - df['ma_20']) / (df['atr'] + 1e-10)
        df['overstretched'] = ((df['rsi'] > 80) | (df['rsi'] < 20) | (df['price_distance_atr'] > 3.0)).astype(int)
        return df

    @staticmethod
    def calculate_chandelier_exit(df, period=22, multiplier=3):
        df = df.copy()
        if 'atr' not in df.columns:
            tr = np.maximum(df['high'] - df['low'],
                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                            abs(df['low']  - df['close'].shift(1))))
            df['atr'] = tr.rolling(period).mean()
        df['highest_high']         = df['high'].rolling(period).max()
        df['lowest_low']           = df['low'].rolling(period).min()
        df['chandelier_long_exit'] = df['highest_high'] - multiplier * df['atr']
        df['chandelier_short_exit']= df['lowest_low']  + multiplier * df['atr']
        return df

    @staticmethod
    def add_time_features(df):
        df = df.copy()
        df['hour']            = df['datetime'].dt.hour
        df['minute']          = df['datetime'].dt.minute
        df['day_of_week']     = df['datetime'].dt.dayofweek
        df['week_of_month']   = (df['datetime'].dt.day - 1) // 7 + 1
        df['day_of_month']    = df['datetime'].dt.day
        df['days_in_month']   = df['datetime'].dt.days_in_month
        df['days_to_month_end'] = df['days_in_month'] - df['day_of_month']
        df['is_expiry_week']  = (df['days_to_month_end'] <= 7).astype(int)
        df['is_opening_hour'] = (df['hour'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour'] == 15).astype(int)
        df['is_mid_session']  = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        return df

    @staticmethod
    def add_technical_indicators(df):
        df = df.copy()
        for p in [5, 10, 20, 50, 200]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['price_vs_sma20']  = (df['close'] - df['sma_20'])  / (df['sma_20']  + 1e-10)
        df['price_vs_sma50']  = (df['close'] - df['sma_50'])  / (df['sma_50']  + 1e-10)
        df['ema_5_20_cross']  = (df['ema_5']  > df['ema_20']).astype(int)
        df['ema_20_50_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
        for p in [5, 10, 20]:
            df[f'roc_{p}'] = ((df['close'] - df['close'].shift(p)) / (df['close'].shift(p) + 1e-10)) * 100
        df['macd']        = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']
        df['volume_sma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)
        df['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-10)
        df['candle_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['candle_body']  = abs(df['close'] - df['open']) / (df['close'] + 1e-10)
        df['upper_wick']   = (df['high'] - np.maximum(df['open'], df['close'])) / (df['close'] + 1e-10)
        df['lower_wick']   = (np.minimum(df['open'], df['close']) - df['low'])   / (df['close'] + 1e-10)
        df['is_bullish']   = (df['close'] > df['open']).astype(int)
        df['returns']      = df['close'].pct_change()
        df['log_returns']  = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
        df['returns_std_5']  = df['returns'].rolling(5).std()
        df['returns_std_20'] = df['returns'].rolling(20).std()
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}']  = df['volume'].shift(lag)
        num_cols = [c for c in df.columns if c not in ['datetime','date','time','open','high','low','close','volume']
                    and df[c].dtype in ['float64','float32','int64','int32']]
        for col in num_cols:
            df[col] = df[col].ffill().fillna(0)
        return df

    @staticmethod
    def create_target_variable(df, horizon=5):
        df = df.copy()
        df['future_return']    = df['close'].shift(-horizon) / df['close'] - 1
        df['future_direction'] = (df['future_return'] > 0).astype(int)
        df['future_high']      = df['high'].rolling(horizon).max().shift(-horizon)
        df['future_low']       = df['low'].rolling(horizon).min().shift(-horizon)
        return df

    def engineer_all_features(self, df, target_horizon=5):
        print("\nEngineering features...")
        df = self.calculate_fibonacci_levels(df);  print("  [OK] Fibonacci levels")
        df = self.identify_fvg(df);                print("  [OK] FVG")
        df = self.calculate_market_regime(df);     print("  [OK] Market regime")
        df = self.detect_overstretched_move(df);   print("  [OK] Overstretched detection")
        df = self.calculate_chandelier_exit(df);   print("  [OK] Chandelier exit")
        df = self.add_time_features(df);           print("  [OK] Time features")
        df = self.add_technical_indicators(df);    print("  [OK] Technical indicators")
        df = self.create_target_variable(df, target_horizon); print("  [OK] Target variable")
        print(f"  Total columns: {len(df.columns)}")
        return df

# ==============================================================================
# PART 3: DATASET PREPARATION
# ==============================================================================

class DatasetPreparer:

    @staticmethod
    def get_feature_columns():
        # Trimmed feature set. Volume features dropped: this dataset has volume=0
        # (FINNIFTY index has no volume), so they're pure noise.
        return [
            'distance_from_ote','in_ote_zone','in_premium_zone','in_discount_zone',
            'in_fvg','bullish_fvg','bearish_fvg','fvg_size',
            'adx','plus_di','minus_di',
            'regime_ranging','regime_trending','regime_strong_up','regime_strong_down',
            'rsi','bb_position','price_distance_atr','overstretched',
            'price_vs_sma20','price_vs_sma50','ema_5_20_cross','ema_20_50_cross',
            'roc_5','roc_10','roc_20','macd_hist',
            'candle_range','candle_body','upper_wick','lower_wick','is_bullish',
            'hour','day_of_week','days_to_month_end',
            'is_expiry_week','is_opening_hour','is_closing_hour','is_mid_session',
            'returns_lag_1','returns_lag_5',
            'bb_std','returns_std_5','returns_std_20',
        ]

    @staticmethod
    def prepare_ml_dataset(df, feature_cols=None):
        if feature_cols is None:
            feature_cols = DatasetPreparer.get_feature_columns()
        target_col   = 'future_direction'
        metadata_cols = ['datetime','open','high','low','close','volume',
                         'in_ote_zone','in_fvg','market_regime',
                         'chandelier_long_exit','chandelier_short_exit',
                         'future_return','future_high','future_low',
                         'bullish_fvg','bearish_fvg','adx','rsi','overstretched',
                         'volume_ratio','atr','plus_di','minus_di']

        available = [c for c in feature_cols if c in df.columns]
        missing   = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"  Warning: {len(missing)} features missing: {missing[:5]} ...")

        col_list = list(dict.fromkeys(available + [target_col] + metadata_cols))
        df_sub   = df[[c for c in col_list if c in df.columns]].copy()
        df_clean = df_sub.dropna(subset=[target_col])

        for col in available:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        threshold = int(len(available) * 0.5)
        df_clean = df_clean.dropna(subset=available, thresh=threshold)

        for col in available:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)

        df_clean = df_clean.reset_index(drop=True)
        X        = df_clean[available]
        y        = df_clean[target_col]
        meta     = df_clean[[c for c in metadata_cols if c in df_clean.columns]]

        print(f"\n  Samples: {len(X):,} | Features: {len(available)}")
        print(f"  Up: {y.sum():,} ({y.mean()*100:.1f}%) | Down: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
        return X, y, meta, available

# ==============================================================================
# PART 4: MODEL TRAINING
# ==============================================================================

class XGBoostTrainer:

    def __init__(self):
        self.model           = None
        self.scaler          = None
        self.feature_names   = None
        self.best_threshold  = 0.5    # long threshold: prob > this -> long
        self.short_threshold = 0.5    # short threshold: prob < this -> short

    def scale_features(self, X_train, X_val, X_test):
        self.scaler = StandardScaler()
        return (self.scaler.fit_transform(X_train),
                self.scaler.transform(X_val),
                self.scaler.transform(X_test))

    def handle_class_imbalance(self, y_train):
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / n_pos
        print(f"  Class ratio - Neg: {n_neg:,} | Pos: {n_pos:,} | scale_pos_weight: {spw:.3f}")
        return spw

    def train_model(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        print("\n" + "="*70)
        print("TRAINING XGBOOST MODEL")
        print("="*70)
        # Shallower trees + stronger regularization + early stopping to combat the
        # train/val AUC gap seen previously (0.75 -> 0.56).
        params = {
            'objective':            'binary:logistic',
            'eval_metric':          ['logloss', 'auc'],
            'max_depth':            5,
            'learning_rate':        0.02,
            'n_estimators':         1000,
            'min_child_weight':     10,
            'subsample':            0.7,
            'colsample_bytree':     0.7,
            'colsample_bylevel':    0.7,
            'gamma':                0.5,
            'reg_alpha':            3.0,
            'reg_lambda':           5.0,
            'max_delta_step':       1,
            'scale_pos_weight':     scale_pos_weight,
            'random_state':         42,
            'n_jobs':               -1,
            'tree_method':          'hist',
            'early_stopping_rounds': 30,
        }
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), (X_val, y_val)],
                       verbose=50)
        print(f"\n[OK] Training complete | best_iteration={self.model.best_iteration}")
        return self.model

    def find_optimal_threshold(self, X_val, y_val, min_precision=0.62, min_signal_rate=0.02):
        # Find BOTH thresholds — asymmetric. Long: smallest t with precision(prob>t)>=min.
        # Short: largest t with precision_down(prob<t)>=min, where precision_down is the
        # accuracy of predicting Down (class 0).
        proba = self.model.predict_proba(X_val)[:, 1]

        # ── Long-side threshold ──────────────────────────────────────────────
        precs_up, recs_up, thrs_up = precision_recall_curve(y_val, proba)
        chosen = None
        for i, t in enumerate(thrs_up):
            sr = (proba > t).mean()
            if precs_up[i] >= min_precision and sr >= min_signal_rate:
                chosen = (t, precs_up[i], recs_up[i], sr)
                break
        if chosen is None:
            f1 = 2 * (precs_up * recs_up) / (precs_up + recs_up + 1e-10)
            bi = int(np.argmax(f1[:-1]))
            self.best_threshold = float(thrs_up[bi])
            print(f"\n  [LONG FALLBACK] No t met precision>={min_precision}, using F1-best")
            print(f"  Long threshold:  {self.best_threshold:.4f} | precision: {precs_up[bi]:.3f} | recall: {recs_up[bi]:.3f}")
        else:
            t, p, r, sr = chosen
            self.best_threshold = float(t)
            print(f"\n  Long threshold:  {self.best_threshold:.4f} | precision_up: {p:.3f} | recall: {r:.3f} | signal_rate: {sr*100:.2f}%")

        # ── Short-side threshold ─────────────────────────────────────────────
        # Treat "Down" as positive: invert labels + probabilities.
        precs_dn, recs_dn, thrs_dn = precision_recall_curve(1 - y_val, 1.0 - proba)
        chosen_s = None
        for i, t in enumerate(thrs_dn):
            # t here is on (1-proba); fire short when (1-proba) > t  i.e. proba < (1-t)
            sr = ((1.0 - proba) > t).mean()
            if precs_dn[i] >= min_precision and sr >= min_signal_rate:
                chosen_s = (1.0 - t, precs_dn[i], recs_dn[i], sr)  # convert back to upper bound on proba
                break
        if chosen_s is None:
            f1 = 2 * (precs_dn * recs_dn) / (precs_dn + recs_dn + 1e-10)
            bi = int(np.argmax(f1[:-1]))
            self.short_threshold = float(1.0 - thrs_dn[bi])
            print(f"  [SHORT FALLBACK] No t met precision>={min_precision}, using F1-best")
            print(f"  Short threshold: {self.short_threshold:.4f} | precision_dn: {precs_dn[bi]:.3f} | recall: {recs_dn[bi]:.3f}")
        else:
            t_short, p, r, sr = chosen_s
            self.short_threshold = float(t_short)
            print(f"  Short threshold: {self.short_threshold:.4f} | precision_dn: {p:.3f} | recall: {r:.3f} | signal_rate: {sr*100:.2f}%")

        return self.best_threshold, self.short_threshold

    def evaluate_model(self, X, y, name="Dataset"):
        print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
        proba  = self.model.predict_proba(X)[:, 1]
        y_pred = (proba > self.best_threshold).astype(int)
        print(classification_report(y, y_pred, target_names=['Down', 'Up']))
        auc = roc_auc_score(y, proba)
        print(f"ROC-AUC: {auc:.4f}")
        return {'predictions': y_pred, 'probabilities': proba, 'auc': auc}

    def plot_feature_importance(self, top_n=30):
        imp = self.model.feature_importances_
        fi  = pd.DataFrame({'feature': self.feature_names, 'importance': imp}) \
                .sort_values('importance', ascending=False)
        print(f"\nTop {top_n} Features:\n{fi.head(top_n).to_string(index=False)}")

        # Categorize features so the bar colors carry signal-family info.
        def _family(name):
            n = name.lower()
            if 'fvg' in n:                                       return 'FVG / Structure'
            if 'fib' in n or 'ote' in n or 'premium' in n or 'discount' in n: return 'Fibonacci / OTE'
            if 'adx' in n or 'di' in n or 'regime' in n:         return 'Trend / Regime'
            if 'rsi' in n or 'bb_' in n or 'overstretched' in n or 'distance_atr' in n: return 'Mean-Reversion'
            if 'roc' in n or 'macd' in n or 'ema' in n or 'sma' in n or 'price_vs' in n: return 'Momentum / MA'
            if 'candle' in n or 'wick' in n or 'is_bullish' in n: return 'Candle Anatomy'
            if 'hour' in n or 'day' in n or 'session' in n or 'opening' in n or 'closing' in n or 'expiry' in n: return 'Time / Session'
            if 'returns_lag' in n or 'returns_std' in n:         return 'Returns / Vol'
            if 'volume' in n:                                    return 'Volume'
            return 'Other'

        family_colors = {
            'FVG / Structure': COLOR_EQUITY,    # cyan
            'Fibonacci / OTE': '#4fc3f7',       # light blue
            'Trend / Regime':  COLOR_WIN,       # neon green
            'Momentum / MA':   COLOR_PURPLE,
            'Mean-Reversion':  COLOR_ACCENT,    # amber
            'Candle Anatomy':  '#ff9e80',       # warm peach
            'Time / Session':  '#80cbc4',       # teal
            'Returns / Vol':   COLOR_LOSS,
            'Volume':          COLOR_MAGENTA,
            'Other':           COLOR_DIM,
        }

        top = fi.head(top_n).reset_index(drop=True)
        top['family'] = top['feature'].apply(_family)
        colors_arr = top['family'].map(family_colors).values

        fig, ax = plt.subplots(figsize=(11, max(7, top_n * 0.32)))
        y_pos = np.arange(len(top))
        bars = ax.barh(y_pos, top['importance'].values, color=colors_arr,
                       edgecolor='white', linewidth=0.6)

        # Value labels at end of each bar
        max_imp = top['importance'].max()
        for bar, val in zip(bars, top['importance'].values):
            ax.text(val + max_imp * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val * 100:.2f}%', va='center', ha='left',
                    fontsize=8, color=COLOR_TEXT)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top['feature'].values, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max_imp * 1.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        ax.set_xlabel('Importance (gain)')
        ax.set_title(f'XGBoost — Top {top_n} Feature Importances', loc='left', pad=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', visible=False)

        # Legend for families that actually appear
        present = [f for f in family_colors if f in top['family'].unique()]
        handles = [plt.Rectangle((0, 0), 1, 1, color=family_colors[f]) for f in present]
        ax.legend(handles, present, loc='lower right', ncol=2, fontsize=8,
                  title='Signal family', title_fontsize=8.5)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
        plt.close(fig)
        print("[OK] Saved: outputs/feature_importance.png")
        return fi

    def save_model(self):
        self.model.save_model(os.path.join(OUTPUT_DIR, 'xgboost_model.json'))
        with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(OUTPUT_DIR, 'best_threshold.pkl'), 'wb') as f:
            pickle.dump(self.best_threshold, f)
        with open(os.path.join(OUTPUT_DIR, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"[OK] Model artifacts saved to outputs/")

# ==============================================================================
# PART 5: BACKTESTING
# ==============================================================================

# Event-day blocklist: macro shocks our ML can't model, so we sit out.
# Sources: RBI MPC schedule + Union Budget dates (Indian markets). Extend as needed.
_RBI_MPC_DATES = [
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-08", "2025-10-08", "2025-12-05",
]
_BUDGET_DATES = ["2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01"]
DEFAULT_EVENT_DATES = set(pd.to_datetime(_RBI_MPC_DATES + _BUDGET_DATES).date)


class TradingBacktester:

    def __init__(self, initial_capital=100000, risk_per_trade=0.01,
                 commission_pct=0.0003, slippage_pct=0.0001,
                 cooldown_bars=10, time_stop_bars=30,
                 partial_exit_r=1.0, trail_atr_mult=1.5,
                 max_position_pct=0.25,
                 session_start=10.0, session_end=14.5,
                 enable_shorts=True,
                 # ── Batch B: risk halts ──────────────────────────────────
                 daily_dd_stop=-2000,      # halt today if intraday P&L breaches
                 streak_loss_stop=4,       # halt today after N consecutive losses
                 strategy_dd_stop=-5000,   # halt strategy if cumulative DD breaches
                 strategy_dd_cooldown_days=3,  # days to sit out after strategy halt
                 event_dates=None):        # set(date) — fall back to DEFAULT_EVENT_DATES
        self.initial_capital  = initial_capital
        self.risk_per_trade   = risk_per_trade        # fraction of equity risked per trade
        self.commission_pct   = commission_pct
        self.slippage_pct     = slippage_pct
        self.cooldown_bars    = cooldown_bars
        self.time_stop_bars   = time_stop_bars
        self.partial_exit_r   = partial_exit_r        # take partial at this R-multiple
        self.trail_atr_mult   = trail_atr_mult        # ATR-mult trail after partial
        self.max_position_pct = max_position_pct      # cap notional at this % of equity
        self.session_start    = session_start         # decimal hour, e.g. 10.0
        self.session_end      = session_end           # decimal hour, e.g. 14.5
        self.enable_shorts    = enable_shorts
        # Risk-halt config
        self.daily_dd_stop             = daily_dd_stop
        self.streak_loss_stop          = streak_loss_stop
        self.strategy_dd_stop          = strategy_dd_stop
        self.strategy_dd_cooldown_days = strategy_dd_cooldown_days
        self.event_dates  = event_dates if event_dates is not None else DEFAULT_EVENT_DATES
        # Model handles
        self.model = self.scaler = self.feature_names = None
        self.threshold       = 0.5    # long  threshold (prob > this)
        self.short_threshold = 0.5    # short threshold (prob < this)

    def generate_signals(self, X, meta):
        X_scaled      = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Asymmetric thresholds — model's edge on Up and Down is not symmetric.
        long_sig  = probabilities > self.threshold
        short_sig = probabilities < self.short_threshold

        # Structural filters (mirrored by direction)
        confluence   = meta['in_ote_zone'].astype(bool) | meta['in_fvg'].astype(bool)
        adx_strong   = meta['adx'] > 20
        up_trend     = adx_strong & (meta['plus_di']  > meta['minus_di'])
        down_trend   = adx_strong & (meta['minus_di'] > meta['plus_di'])
        not_stretch  = ~meta['overstretched'].astype(bool)

        # Session filter — skip first 45 min (open volatility) and last hour (close noise)
        time_dec     = meta['datetime'].dt.hour + meta['datetime'].dt.minute / 60.0
        in_session   = (time_dec >= self.session_start) & (time_dec <= self.session_end)

        long_filter  = long_sig  & confluence & up_trend   & not_stretch & in_session
        short_filter = short_sig & confluence & down_trend & not_stretch & in_session

        # Encode: +1 long, -1 short, 0 flat
        signals = np.zeros(len(meta), dtype=int)
        signals[long_filter.values]  = 1
        if self.enable_shorts:
            signals[short_filter.values] = -1

        print(f"\n  Raw long signals:  {long_sig.sum():,}  | Raw short signals: {short_sig.sum():,}")
        print(f"  After filter long:  {(signals==1).sum():,}")
        print(f"  After filter short: {(signals==-1).sum():,}")
        print(f"    confluence (OTE | FVG):    {confluence.sum():,}")
        print(f"    up_trend (ADX>20 & +DI):   {up_trend.sum():,}")
        print(f"    down_trend (ADX>20 & -DI): {down_trend.sum():,}")
        print(f"    in_session ({self.session_start}-{self.session_end}): {in_session.sum():,}")
        return signals, probabilities

    def backtest(self, X, meta, signals, probabilities):
        print("\n" + "="*70 + "\nRUNNING BACKTEST\n" + "="*70)
        capital      = self.initial_capital
        equity_curve = [capital]
        trades       = []

        in_position       = False
        direction         = 0             # +1 long, -1 short
        entry_price       = 0.0
        shares            = 0.0
        initial_shares    = 0.0
        stop_loss         = 0.0
        partial_target    = 0.0
        partial_taken     = False
        entry_idx         = 0
        entry_capital     = capital
        extreme_since     = 0.0           # max high (long) or min low (short)
        last_exit_bar     = -10**9
        entry_commission_total = 0.0  # snapshot at entry, charged proportionally to each leg

        # ── Batch B: risk-halt state ─────────────────────────────────────
        current_date         = None
        daily_pnl            = 0.0      # sum of P&L realized today (all legs)
        streak_losses        = 0        # consecutive losing POSITIONS today
        position_pnl_accum   = 0.0      # sum of legs P&L for the currently-open position
        strategy_peak        = capital
        strategy_halt_until  = None     # date — no entries until past this
        # Halt-reason counters for the summary
        halt_counts = {'event': 0, 'daily_dd': 0, 'streak': 0, 'strategy_dd': 0}

        def realize_exit(exit_idx, exit_price_raw, exit_reason, sh, partial=False):
            nonlocal capital, daily_pnl, position_pnl_accum
            ex_price  = exit_price_raw * (1 - self.slippage_pct * direction)
            notional  = sh * ex_price
            exit_commission = notional * self.commission_pct
            # Charge entry commission proportionally to closed shares (relative to
            # the original position size). Sums to exactly entry_commission_total
            # across all exit legs.
            entry_comm_slice = (
                entry_commission_total * (sh / initial_shares)
                if initial_shares > 0 else 0
            )
            pnl = (ex_price - entry_price) * sh * direction - exit_commission - entry_comm_slice
            capital += pnl + entry_comm_slice  # entry slice already off capital at open
            daily_pnl          += pnl
            position_pnl_accum += pnl
            trades.append({
                'entry_time':     meta.iloc[entry_idx]['datetime'],
                'exit_time':      meta.iloc[exit_idx]['datetime'],
                'direction':      'LONG' if direction == 1 else 'SHORT',
                'entry_price':    entry_price,
                'exit_price':     ex_price,
                'shares':         sh,
                'pnl':            pnl,
                'pnl_pct':        pnl / entry_capital * 100,
                'return':         (ex_price / entry_price - 1) * direction,
                'exit_reason':    exit_reason,
                'holding_period': exit_idx - entry_idx,
                'probability':    probabilities[entry_idx],
                'market_regime':  meta.iloc[entry_idx]['market_regime'],
                'partial':        partial,
            })

        def close_position(idx, exit_price_raw, reason):
            """Realize final exit + update streak/strategy-DD state."""
            nonlocal in_position, last_exit_bar, shares, direction
            nonlocal streak_losses, strategy_peak, strategy_halt_until, position_pnl_accum
            realize_exit(idx, exit_price_raw, reason, shares)
            in_position = False; last_exit_bar = idx; shares = 0; direction = 0

            # Update consecutive-loss streak based on the position's total P&L
            if position_pnl_accum < 0:
                streak_losses += 1
            else:
                streak_losses = 0
            position_pnl_accum = 0.0

            # Update strategy peak/DD; trip the multi-day halt if breached
            if capital > strategy_peak:
                strategy_peak = capital
            dd_from_peak = capital - strategy_peak
            if dd_from_peak <= self.strategy_dd_stop:
                strategy_halt_until = current_date + timedelta(days=self.strategy_dd_cooldown_days)
                strategy_peak = capital  # reset so we don't immediately re-trip

        for i in range(len(meta)):
            row   = meta.iloc[i]
            price = row['close']
            high  = row['high']
            low   = row['low']
            this_date = row['datetime'].date() if hasattr(row['datetime'], 'date') else pd.Timestamp(row['datetime']).date()

            # Reset daily counters on date change
            if this_date != current_date:
                current_date  = this_date
                daily_pnl     = 0.0
                streak_losses = 0

            # ── Manage open position ─────────────────────────────────────────
            if in_position:
                atr_now = row['atr'] if pd.notna(row['atr']) else 0.0

                if direction == 1:
                    extreme_since = max(extreme_since, high)
                    if partial_taken and atr_now > 0:
                        trail = extreme_since - self.trail_atr_mult * atr_now
                        stop_loss = max(stop_loss, trail)
                    else:
                        new_stop = row['chandelier_long_exit']
                        if pd.notna(new_stop):
                            stop_loss = max(stop_loss, new_stop)
                else:  # short
                    extreme_since = min(extreme_since, low)
                    if partial_taken and atr_now > 0:
                        trail = extreme_since + self.trail_atr_mult * atr_now
                        stop_loss = min(stop_loss, trail)
                    else:
                        new_stop = row['chandelier_short_exit']
                        if pd.notna(new_stop):
                            stop_loss = min(stop_loss, new_stop)

                bars_held = i - entry_idx

                # Partial take-profit at the R-multiple target
                if not partial_taken:
                    hit_target = (high >= partial_target) if direction == 1 else (low <= partial_target)
                    if hit_target:
                        half_sh = initial_shares * 0.5
                        realize_exit(i, partial_target, f"Partial TP ({self.partial_exit_r}R)", half_sh, partial=True)
                        shares -= half_sh
                        partial_taken = True
                        # move stop to break-even after partial
                        if direction == 1:
                            stop_loss = max(stop_loss, entry_price)
                        else:
                            stop_loss = min(stop_loss, entry_price)

                # Hard stop (intrabar pierce)
                stop_hit = (low <= stop_loss) if direction == 1 else (high >= stop_loss)
                in_profit = (price > entry_price) if direction == 1 else (price < entry_price)

                if stop_hit:
                    close_position(i, stop_loss, "Stop Loss")
                elif bars_held >= self.time_stop_bars and not in_profit:
                    close_position(i, price, "Time Stop")
                elif i == len(meta) - 1:
                    close_position(i, price, "End of Data")

            # ── Look for new entry ───────────────────────────────────────────
            halt_reason = None
            entry_candidate = (
                (not in_position)
                and signals[i] != 0
                and (i - last_exit_bar) > self.cooldown_bars
            )
            if entry_candidate:
                if current_date in self.event_dates:
                    halt_reason = 'event'
                elif strategy_halt_until is not None and current_date <= strategy_halt_until:
                    halt_reason = 'strategy_dd'
                elif daily_pnl <= self.daily_dd_stop:
                    halt_reason = 'daily_dd'
                elif streak_losses >= self.streak_loss_stop:
                    halt_reason = 'streak'

            if entry_candidate and halt_reason is not None:
                halt_counts[halt_reason] += 1
            elif entry_candidate:
                sig = signals[i]
                # slippage costs in direction of opening trade
                #   buying long  -> pay up
                #   selling short -> sell down
                entry_price_candidate = price * (1 + self.slippage_pct * sig)
                if sig == 1:
                    init_stop = row['chandelier_long_exit']
                    stop_valid = pd.notna(init_stop) and init_stop < entry_price_candidate
                    risk_per_share = (entry_price_candidate - init_stop) if stop_valid else 0
                else:  # short
                    init_stop = row['chandelier_short_exit']
                    stop_valid = pd.notna(init_stop) and init_stop > entry_price_candidate
                    risk_per_share = (init_stop - entry_price_candidate) if stop_valid else 0

                if stop_valid and risk_per_share > 0:
                    dollar_risk  = capital * self.risk_per_trade
                    sh           = dollar_risk / risk_per_share
                    notional     = sh * entry_price_candidate
                    cap_notional = capital * self.max_position_pct
                    if notional > cap_notional:
                        sh       = cap_notional / entry_price_candidate
                        notional = sh * entry_price_candidate
                    if sh > 0:
                        entry_commission = notional * self.commission_pct
                        capital         -= entry_commission
                        entry_capital    = capital
                        shares           = sh
                        initial_shares   = sh
                        entry_price      = entry_price_candidate
                        stop_loss        = init_stop
                        direction        = sig
                        partial_target   = entry_price + sig * self.partial_exit_r * risk_per_share
                        partial_taken    = False
                        extreme_since    = high if sig == 1 else low
                        in_position      = True
                        entry_idx        = i
                        # remember entry commission so exits can amortize it into reported PnL
                        entry_commission_total = entry_commission

            # ── Mark-to-market equity ────────────────────────────────────────
            if in_position:
                equity_curve.append(capital + (price - entry_price) * shares * direction)
            else:
                equity_curve.append(capital)

        trades_df = pd.DataFrame(trades)
        if len(trades_df):
            trades_df['profitable'] = (trades_df['pnl'] > 0).astype(int)

        final_return = (capital - self.initial_capital) / self.initial_capital * 100
        n_entries = len(trades_df[~trades_df['partial']]) if len(trades_df) else 0
        n_long  = (trades_df['direction'] == 'LONG').sum()  if len(trades_df) else 0
        n_short = (trades_df['direction'] == 'SHORT').sum() if len(trades_df) else 0
        print(f"  Trade legs: {len(trades_df)} (entries: {n_entries} | long legs: {n_long} | short legs: {n_short})")
        print(f"  Final capital: ${capital:,.2f} | Return: {final_return:.2f}%")
        total_halts = sum(halt_counts.values())
        if total_halts > 0:
            print(f"  Halted entries: {total_halts}  "
                  f"(event:{halt_counts['event']}  "
                  f"daily_dd:{halt_counts['daily_dd']}  "
                  f"streak:{halt_counts['streak']}  "
                  f"strategy_dd:{halt_counts['strategy_dd']})")
        return trades_df, np.array(equity_curve)

# ==============================================================================
# PART 6: PERFORMANCE METRICS
# ==============================================================================

class PerformanceEvaluator:

    @staticmethod
    def calculate_sharpe(returns):
        if len(returns) == 0 or returns.std() == 0: return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_sortino(returns):
        down = returns[returns < 0]
        if len(down) == 0 or down.std() == 0: return 0.0
        return returns.mean() / down.std() * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(equity):
        eq  = np.array(equity)
        peak= np.maximum.accumulate(eq)
        return ((eq - peak) / peak).min()

    @staticmethod
    def generate_report(trades_df, equity_curve, initial_capital):
        print("\n" + "="*70 + "\nPERFORMANCE REPORT\n" + "="*70)
        if len(trades_df) == 0:
            print("No trades executed."); return {}

        returns = trades_df['return'].values
        sharpe  = PerformanceEvaluator.calculate_sharpe(pd.Series(returns))
        sortino = PerformanceEvaluator.calculate_sortino(pd.Series(returns))
        max_dd  = PerformanceEvaluator.calculate_max_drawdown(equity_curve)
        total_r = (equity_curve[-1] - initial_capital) / initial_capital * 100

        wins       = trades_df[trades_df['pnl'] > 0]
        losses     = trades_df[trades_df['pnl'] < 0]
        win_rate   = len(wins) / len(trades_df) * 100
        gross_p    = wins['pnl'].sum()
        gross_l    = abs(losses['pnl'].sum())
        pf         = gross_p / gross_l if gross_l > 0 else float('inf')
        avg_win    = wins['pnl'].mean()   if len(wins)   else 0
        avg_loss   = losses['pnl'].mean() if len(losses) else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        print(f"\n  RISK-ADJUSTED")
        print(f"  Sharpe:         {sharpe:>10.4f}")
        print(f"  Sortino:        {sortino:>10.4f}")
        print(f"  Max Drawdown:   {max_dd*100:>10.2f}%")
        print(f"\n  TRADE STATS")
        print(f"  Total Trades:   {len(trades_df):>10,}")
        print(f"  Win Rate:       {win_rate:>10.2f}%")
        print(f"  Profit Factor:  {pf:>10.2f}")
        print(f"  Expectancy:     ${expectancy:>10,.2f}")
        print(f"\n  RETURNS")
        print(f"  Total Return:   {total_r:>10.2f}%")
        print(f"  Gross Profit:   ${gross_p:>10,.2f}")
        print(f"  Gross Loss:     ${gross_l:>10,.2f}")

        return {'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
                'total_return_pct': total_r, 'win_rate': win_rate,
                'profit_factor': pf, 'expectancy': expectancy,
                'total_trades': len(trades_df)}

# ==============================================================================
# PART 7: VISUALIZATION
# ==============================================================================

def _currency_fmt(x, _):
    if abs(x) >= 1_000_000: return f'Rs.{x/1e6:.2f}M'
    if abs(x) >= 1_000:     return f'Rs.{x/1e3:.0f}k'
    return f'Rs.{x:,.0f}'

def _spot_fmt(x, _):
    return f'{x:,.0f}'

def _pct_fmt(x, _):
    return f'{x:.1f}%'

def _style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, color='#cfcfcf')

# ─────────────────────────────────────────────────────────────────────────────
# Each function saves ONE focused PNG. Smooth lines (trade-# axis where
# applicable), white background, muted institutional palette.
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    out = os.path.join(OUTPUT_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Saved: {out}")

def plot_equity(trades_df, initial_capital, label='Test'):
    """Cumulative P&L + drawdown on a real datetime axis."""
    if len(trades_df) == 0:
        print("[skip] equity plot — no trades"); return

    # Sort by realization time so the curve is chronological.
    td = trades_df.copy()
    td['exit_time'] = pd.to_datetime(td['exit_time'])
    td = td.sort_values('exit_time').reset_index(drop=True)

    ts         = td['exit_time']
    cum_pnl    = td['pnl'].cumsum().values
    equity     = initial_capital + cum_pnl
    peak       = np.maximum.accumulate(equity)
    dd_abs     = equity - peak
    dd_pct     = (equity - peak) / peak * 100
    max_dd_abs = dd_abs.min()
    max_dd_pct = dd_pct.min()
    final_pnl  = cum_pnl[-1]
    total_ret  = final_pnl / initial_capital * 100
    pnl_color  = COLOR_WIN if final_pnl >= 0 else COLOR_LOSS

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.20)

    # Top: cumulative P&L
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(ts, cum_pnl, 0, where=cum_pnl >= 0,
                     color=COLOR_WIN, alpha=0.15, linewidth=0)
    ax1.fill_between(ts, cum_pnl, 0, where=cum_pnl < 0,
                     color=COLOR_LOSS, alpha=0.15, linewidth=0)
    ax1.plot(ts, cum_pnl, color=pnl_color, linewidth=2.2)
    ax1.axhline(0, color=COLOR_DIM, linewidth=0.8, linestyle=':')

    # Headline annotation at the final point — arrow pointing to the end value
    ax1.annotate(
        f'Rs.{final_pnl:+,.0f}  ({total_ret:+.2f}%)',
        xy=(ts.iloc[-1], final_pnl),
        xytext=(-90, 18), textcoords='offset points',
        fontsize=11, weight='bold', color=pnl_color,
        ha='right', va='center',
        arrowprops=dict(arrowstyle='->', color=pnl_color, lw=1.4),
    )

    ax1.set_title(f'{label}  —  Cumulative P&L', loc='left', pad=10, color=COLOR_TEXT)
    ax1.set_ylabel('Cumulative P&L (Rs.)')
    ax1.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    plt.setp(ax1.get_xticklabels(), visible=False)
    _style_axis(ax1)

    # Bottom: drawdown — shares the datetime x-axis
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(ts, dd_abs, 0, color=COLOR_DRAWDOWN, alpha=0.30, linewidth=0)
    ax2.plot(ts, dd_abs, color=COLOR_DRAWDOWN, linewidth=1.6)
    ax2.axhline(0, color=COLOR_DIM, linewidth=0.7)

    # Annotation at the worst drawdown point
    trough_idx = int(np.argmin(dd_abs))
    ax2.annotate(
        f'Max DD  Rs.{max_dd_abs:,.0f}  ({max_dd_pct:.2f}%)',
        xy=(ts.iloc[trough_idx], dd_abs[trough_idx]),
        xytext=(15, -8), textcoords='offset points',
        fontsize=9.5, weight='bold', color=COLOR_DRAWDOWN,
        ha='left', va='top',
        arrowprops=dict(arrowstyle='->', color=COLOR_DRAWDOWN, lw=1.2),
    )

    ax2.set_title('Drawdown', loc='left', pad=6, color=COLOR_TEXT)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (Rs.)')
    ax2.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    _style_axis(ax2)
    fig.autofmt_xdate()

    _save(fig, f'{label.lower().replace(" ", "_")}_equity.png')


def plot_finnifty_spot(test_meta, trades_df, label='Test'):
    """FINNIFTY spot price chart with entry markers."""
    if len(test_meta) == 0:
        print("[skip] spot plot — no meta"); return
    times  = pd.to_datetime(test_meta['datetime'])
    prices = test_meta['close'].values

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, prices, color=COLOR_SPOT, linewidth=1.1, label='FINNIFTY close')

    # Overlay entries (green up-tri for long, red down-tri for short)
    if len(trades_df):
        entries = trades_df[~trades_df.get('partial', pd.Series([False]*len(trades_df)))]
        if 'partial' in trades_df.columns:
            entries = trades_df[~trades_df['partial']]
        else:
            entries = trades_df
        entries = entries.copy()
        entries['entry_time'] = pd.to_datetime(entries['entry_time'])

        longs  = entries[entries.get('direction', 'LONG') == 'LONG']  if 'direction' in entries.columns else entries
        shorts = entries[entries.get('direction', 'LONG') == 'SHORT'] if 'direction' in entries.columns else entries.iloc[0:0]

        if len(longs):
            ax.scatter(longs['entry_time'], longs['entry_price'],
                       marker='^', color=COLOR_WIN, s=42,
                       edgecolor='white', linewidth=0.6, zorder=5,
                       label=f'Long entry ({len(longs)})')
        if len(shorts):
            ax.scatter(shorts['entry_time'], shorts['entry_price'],
                       marker='v', color=COLOR_LOSS, s=42,
                       edgecolor='white', linewidth=0.6, zorder=5,
                       label=f'Short entry ({len(shorts)})')

    ax.set_title(f'FINNIFTY  —  Spot Price ({label})',
                 loc='left', pad=10, color=COLOR_TEXT)
    ax.set_ylabel('Spot price')
    ax.set_xlabel('Date')
    ax.yaxis.set_major_formatter(FuncFormatter(_spot_fmt))
    ax.legend(loc='upper left')
    fig.autofmt_xdate()
    _style_axis(ax)

    _save(fig, f'{label.lower().replace(" ", "_")}_spot.png')


def plot_trade_quality(trades_df, label='Test'):
    """3-panel: P&L distribution | Exit reasons | Holding period."""
    if len(trades_df) == 0: return
    pnls   = trades_df['pnl'].values
    wins   = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                            width_ratios=[1, 1, 1])

    # P&L distribution
    ax1  = fig.add_subplot(gs[0])
    bins = max(15, min(30, len(pnls) // 3))
    _, edges, patches = ax1.hist(pnls, bins=bins, edgecolor='white', linewidth=0.5)
    for patch, edge_left in zip(patches, edges[:-1]):
        patch.set_facecolor(COLOR_WIN if edge_left >= 0 else COLOR_LOSS)
        patch.set_alpha(0.85)
    ax1.axvline(0, color=COLOR_DIM, linewidth=0.8)
    ax1.axvline(pnls.mean(), color=COLOR_ACCENT, linewidth=1.6, linestyle='--',
                label=f'Mean Rs.{pnls.mean():,.0f}')
    ax1.set_title('P&L Distribution', loc='left', pad=8)
    ax1.set_xlabel('P&L per trade leg')
    ax1.set_ylabel('Count')
    ax1.xaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_WIN, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_LOSS, alpha=0.85),
        plt.Line2D([0], [0], color=COLOR_ACCENT, linewidth=1.6, linestyle='--'),
    ]
    ax1.legend(legend_handles,
               [f'Win  ({len(wins)})', f'Loss ({len(losses)})',
                f'Mean Rs.{pnls.mean():,.0f}'],
               loc='upper right')
    _style_axis(ax1)

    # Exit reasons (horizontal bar with avg P&L + count)
    ax2 = fig.add_subplot(gs[1])
    by_reason = trades_df.groupby('exit_reason').agg(
        total_pnl=('pnl', 'sum'),
        count=('pnl', 'size'),
    ).sort_values('total_pnl')
    y_pos    = np.arange(len(by_reason))
    bar_clrs = [COLOR_WIN if v >= 0 else COLOR_LOSS for v in by_reason['total_pnl']]
    ax2.barh(y_pos, by_reason['total_pnl'].values, color=bar_clrs,
             edgecolor='white', linewidth=0.6)
    span = abs(by_reason['total_pnl']).max() if len(by_reason) else 1
    for i, (v, c) in enumerate(zip(by_reason['total_pnl'], by_reason['count'])):
        ha = 'left' if v >= 0 else 'right'
        off = span * 0.02
        ax2.text(v + (off if v >= 0 else -off), i,
                 f' n={c}', va='center', ha=ha, fontsize=9, color=COLOR_TEXT)
    ax2.axvline(0, color=COLOR_DIM, linewidth=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(by_reason.index)
    ax2.set_title('P&L by Exit Reason', loc='left', pad=8)
    ax2.set_xlabel('Total P&L')
    ax2.xaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    _style_axis(ax2)
    ax2.grid(axis='y', visible=False)

    # Holding period
    ax3 = fig.add_subplot(gs[2])
    hp  = trades_df['holding_period'].values
    ax3.hist(hp, bins=max(10, min(25, len(hp) // 3)),
             color=COLOR_EQUITY, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax3.axvline(np.mean(hp),   color=COLOR_ACCENT,  linewidth=1.6, linestyle='--',
                label=f'Mean   {np.mean(hp):.0f} bars')
    ax3.axvline(np.median(hp), color=COLOR_NEUTRAL, linewidth=1.4, linestyle=':',
                label=f'Median {np.median(hp):.0f} bars')
    ax3.set_title('Holding Period', loc='left', pad=8)
    ax3.set_xlabel('Bars per leg')
    ax3.set_ylabel('Count')
    ax3.legend(loc='upper right')
    _style_axis(ax3)

    _save(fig, f'{label.lower().replace(" ", "_")}_trade_quality.png')


def plot_time_analysis(trades_df, label='Test'):
    """4-panel: Monthly P&L | Weekday P&L | Hour P&L | Rolling win rate."""
    if len(trades_df) == 0: return
    td = trades_df.copy()
    et = pd.to_datetime(td['entry_time'])
    td['_hour']  = et.dt.hour
    td['_dow']   = et.dt.day_name().str[:3]
    td['_month'] = et.dt.to_period('M').astype(str)

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28)

    # Monthly P&L
    ax1 = fig.add_subplot(gs[0, :])      # full top row — months span wide
    by_month = td.groupby('_month')['pnl'].sum().sort_index()
    colors_m = [COLOR_WIN if v >= 0 else COLOR_LOSS for v in by_month.values]
    ax1.bar(range(len(by_month)), by_month.values, color=colors_m,
            edgecolor='white', linewidth=0.6)
    ax1.axhline(0, color=COLOR_DIM, linewidth=0.7)
    ax1.set_xticks(range(len(by_month)))
    ax1.set_xticklabels(by_month.index, rotation=35, ha='right')
    ax1.set_title('Monthly P&L', loc='left', pad=8)
    ax1.set_ylabel('Net P&L')
    ax1.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    _style_axis(ax1)
    ax1.grid(axis='x', visible=False)

    # Weekday
    ax2 = fig.add_subplot(gs[1, 0])
    dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    by_dow = td.groupby('_dow')['pnl'].sum().reindex(dow_order, fill_value=0)
    colors_d = [COLOR_WIN if v >= 0 else COLOR_LOSS for v in by_dow.values]
    ax2.bar(by_dow.index, by_dow.values, color=colors_d,
            edgecolor='white', linewidth=0.6)
    ax2.axhline(0, color=COLOR_DIM, linewidth=0.7)
    ax2.set_title('P&L by Weekday', loc='left', pad=8)
    ax2.set_ylabel('Net P&L')
    ax2.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    _style_axis(ax2)
    ax2.grid(axis='x', visible=False)

    # Hour of day
    ax3 = fig.add_subplot(gs[1, 1])
    by_hr = td.groupby('_hour')['pnl'].sum()
    all_h = list(range(by_hr.index.min(), by_hr.index.max() + 1))
    by_hr = by_hr.reindex(all_h, fill_value=0)
    colors_h = [COLOR_WIN if v >= 0 else COLOR_LOSS for v in by_hr.values]
    ax3.bar(by_hr.index, by_hr.values, color=colors_h,
            edgecolor='white', linewidth=0.6)
    ax3.axhline(0, color=COLOR_DIM, linewidth=0.7)
    ax3.set_title('P&L by Entry Hour', loc='left', pad=8)
    ax3.set_xticks(list(by_hr.index))
    ax3.set_xlabel('Hour of day')
    ax3.set_ylabel('Net P&L')
    ax3.yaxis.set_major_formatter(FuncFormatter(_currency_fmt))
    _style_axis(ax3)
    ax3.grid(axis='x', visible=False)

    _save(fig, f'{label.lower().replace(" ", "_")}_time_analysis.png')


def plot_results(equity_curve, trades_df, label='Test', initial_capital=None,
                 test_meta=None):
    """Top-level: render the full report as separate PNGs."""
    if initial_capital is None:
        initial_capital = equity_curve[0] if len(equity_curve) else 100000

    plot_equity(trades_df, initial_capital, label)
    if test_meta is not None:
        plot_finnifty_spot(test_meta, trades_df, label)
    plot_trade_quality(trades_df, label)
    plot_time_analysis(trades_df, label)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FINNIFTY ALGORITHMIC TRADING SYSTEM")
    print("="*70)

    # ── 1. Load & engineer features ───────────────────────────────────────────
    loader   = DataLoader()
    engineer = FeatureEngineer()
    preparer = DatasetPreparer()

    train_raw, val_raw, test_raw = loader.split_train_val_test()

    print("\n--- Feature Engineering ---")
    train_df = engineer.engineer_all_features(train_raw)
    val_df   = engineer.engineer_all_features(val_raw)
    test_df  = engineer.engineer_all_features(test_raw)

    print("\n--- Preparing Datasets ---")
    X_train, y_train, train_meta, feat_cols = preparer.prepare_ml_dataset(train_df)
    X_val,   y_val,   val_meta,   _         = preparer.prepare_ml_dataset(val_df,   feat_cols)
    X_test,  y_test,  test_meta,  _         = preparer.prepare_ml_dataset(test_df,  feat_cols)

    if len(X_train) == 0:
        raise ValueError("Training set is empty — check CSV paths and data.")

    if len(X_val) == 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        val_meta = train_meta.iloc[:len(X_val)].copy()

    if len(X_test) == 0:
        X_test, y_test, test_meta = X_val, y_val, val_meta

    print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── 2. Train model ────────────────────────────────────────────────────────
    trainer = XGBoostTrainer()
    trainer.feature_names = list(X_train.columns)

    X_tr_s, X_v_s, X_te_s = trainer.scale_features(X_train, X_val, X_test)
    spw   = trainer.handle_class_imbalance(y_train)
    model = trainer.train_model(X_tr_s, y_train, X_v_s, y_val, spw)
    trainer.find_optimal_threshold(X_v_s, y_val)
    trainer.evaluate_model(X_v_s,  y_val,  "Validation Set")
    trainer.evaluate_model(X_te_s, y_test, "Test Set")
    trainer.plot_feature_importance(top_n=30)
    trainer.save_model()

    # ── 3. Backtest ───────────────────────────────────────────────────────────
    backtester = TradingBacktester(
        initial_capital=100000,
        risk_per_trade=0.01,        # 1% equity risked per trade
        commission_pct=0.00007,     # ~0.007% per side — realistic for FINNIFTY futures
        slippage_pct=0.0001,
        cooldown_bars=10,           # no re-entry for 10 bars after exit
        time_stop_bars=30,          # bail if not profitable in 30 bars
        partial_exit_r=1.0,         # take 50% off at +1R
        trail_atr_mult=1.5,         # ATR-trail on the runner after partial
        max_position_pct=10.0,      # allow up to 10x leverage (futures-realistic)
        session_start=10.0,         # skip first 45 min of the day
        session_end=14.5,           # skip last hour
        enable_shorts=False,        # longs only — model has weak short precision
        # Risk halts (Batch B)
        daily_dd_stop=-2000,        # halt today if intraday P&L breaches -Rs.2,000
        streak_loss_stop=4,         # halt today after 4 consecutive losing positions
        strategy_dd_stop=-5000,     # halt strategy if cumulative DD breaches -Rs.5,000
        strategy_dd_cooldown_days=3,# sit out 3 days after a strategy DD halt
        # event_dates=DEFAULT_EVENT_DATES (RBI MPC + Budget) — leave default
    )
    backtester.model            = model
    backtester.scaler           = trainer.scaler
    backtester.threshold        = trainer.best_threshold
    backtester.short_threshold  = trainer.short_threshold
    backtester.feature_names    = feat_cols

    signals, probs     = backtester.generate_signals(X_test, test_meta)
    trades_df, eq_curve= backtester.backtest(X_test, test_meta, signals, probs)

    # ── 4. Performance ────────────────────────────────────────────────────────
    evaluator = PerformanceEvaluator()
    metrics   = evaluator.generate_report(trades_df, eq_curve, backtester.initial_capital)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    if len(trades_df):
        trades_df.to_csv(os.path.join(OUTPUT_DIR, 'trade_log.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, 'metrics.csv'), index=False)
    print(f"\n[OK] Results saved to outputs/")

    plot_results(eq_curve, trades_df, "Test Set",
                 initial_capital=backtester.initial_capital,
                 test_meta=test_meta)

    print("\n" + "="*70)
    print("  DONE — check the outputs/ folder for all results")
    print("="*70)
