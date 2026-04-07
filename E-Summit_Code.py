"""
FINNIFTY COMPLETE ALGORITHMIC TRADING SYSTEM
Integrated: Data Processing + Feature Engineering + Model Training + Backtesting
Structure-based Trading with OTE (Fibonacci 0.5-0.7) + FVG + Market Regime
ML Model: XGBoost with optimized hyperparameters
Exit Strategy: Chandelier Stop + Take Profit
"""
!pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import warnings
import glob
import pickle
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

# Plotting setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#=============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
#=============================================================================

class DataLoader:
    """Load and preprocess FINNIFTY data"""

    def __init__(self, base_path='/content/drive/MyDrive/FINNIFTY'):
        self.base_path = base_path

    def load_data(self, file_indices=None):
        """
        Load FINNIFTY CSV files

        Parameters:
        -----------
        file_indices : list or None
            Specific file indices to load (e.g., [0,1,2] for first 3 files)
            If None, loads all files
        """
        all_files = sorted(glob.glob(f"{self.base_path}/*.csv"))

        if file_indices is not None:
            all_files = [all_files[i] for i in file_indices if i < len(all_files)]

        print(f"Loading {len(all_files)} FINNIFTY files...")

        dfs = []
        for file in tqdm(all_files):
            df = pd.read_csv(file)
            # Clean date format (remove Excel formula artifacts)
            df['date'] = df['date'].astype(str).str.replace('="', '').str.replace('"', '')
            dfs.append(df)

        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)

        # Parse datetime
        combined_df['datetime'] = pd.to_datetime(
            combined_df['date'] + ' ' + combined_df['time'],
            format='%d-%m-%y %H:%M:%S',
            errors='coerce'
        )

        # Remove any rows with invalid datetime
        combined_df = combined_df.dropna(subset=['datetime'])

        # Sort by datetime
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

        # Filter market hours only (9:15 AM - 3:30 PM)
        combined_df['hour'] = combined_df['datetime'].dt.hour
        combined_df['minute'] = combined_df['datetime'].dt.minute
        combined_df['time_decimal'] = combined_df['hour'] + combined_df['minute'] / 60

        # Market hours: 9:15 (9.25) to 15:30 (15.5)
        market_hours_mask = (
            (combined_df['time_decimal'] >= 9.25) &
            (combined_df['time_decimal'] <= 15.5)
        )
        combined_df = combined_df[market_hours_mask].reset_index(drop=True)

        print(f"Loaded {len(combined_df):,} records from {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

        return combined_df

    def split_train_val_test(self):
        """
        Split data into train, validation, and test sets

        Returns:
        --------
        train_df, val_df, test_df
        """
        print("\n" + "="*80)
        print("SPLITTING DATA: TRAIN / VALIDATION / TEST")
        print("="*80)

        # Load different sets
        train_df = self.load_data(file_indices=list(range(0, 12)))  # Files 1-12
        val_df = self.load_data(file_indices=list(range(12, 14)))   # Files 13-14
        test_df = self.load_data(file_indices=list(range(14, 17)))  # Files 15-17

        print(f"\nTrain set: {len(train_df):,} records ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        print(f"Validation set: {len(val_df):,} records ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        print(f"Test set: {len(test_df):,} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

        return train_df, val_df, test_df

#=============================================================================
# PART 2: ENHANCED FEATURE ENGINEERING
#=============================================================================

class FeatureEngineer:
    """Engineer trading features based on market structure"""

    @staticmethod
    def calculate_fibonacci_levels(df, lookback=100):
        """
        Calculate Fibonacci retracement levels
        ENHANCED: OTE zone now covers 0.5-0.7 for better entries
        """
        df = df.copy()

        # Calculate swing high and low over lookback period
        df['swing_high_price'] = df['high'].rolling(window=lookback).max()
        df['swing_low_price'] = df['low'].rolling(window=lookback).min()

        # Calculate Fibonacci levels
        df['fib_range'] = df['swing_high_price'] - df['swing_low_price']
        df['fib_0.236'] = df['swing_high_price'] - 0.236 * df['fib_range']
        df['fib_0.382'] = df['swing_high_price'] - 0.382 * df['fib_range']
        df['fib_0.500'] = df['swing_high_price'] - 0.500 * df['fib_range']  # OTE start
        df['fib_0.618'] = df['swing_high_price'] - 0.618 * df['fib_range']
        df['fib_0.700'] = df['swing_high_price'] - 0.700 * df['fib_range']  # OTE end
        df['fib_0.786'] = df['swing_high_price'] - 0.786 * df['fib_range']

        # ENHANCED: OTE zone is now 0.5 to 0.7 (broader range for more entries)
        # Allow 1% tolerance
        tolerance = df['fib_range'] * 0.01
        df['in_ote_zone'] = (
            (df['close'] >= (df['fib_0.700'] - tolerance)) &
            (df['close'] <= (df['fib_0.500'] + tolerance))
        )

        # Distance from OTE center (0.6)
        df['fib_0.600'] = df['swing_high_price'] - 0.600 * df['fib_range']
        df['distance_from_ote'] = (df['close'] - df['fib_0.600']) / df['fib_range']

        # Additional Fib features
        df['in_premium_zone'] = (df['close'] > df['fib_0.500']).astype(int)  # Above 50%
        df['in_discount_zone'] = (df['close'] < df['fib_0.500']).astype(int)  # Below 50%

        return df

    @staticmethod
    def identify_fvg(df, min_gap_pct=0.05):
        """
        Identify Fair Value Gaps (FVG)
        ENHANCED: Reduced min gap to 0.05% for more signals
        """
        df = df.copy()

        # Bullish FVG: gap up (candle 1 high < candle 3 low)
        df['bullish_fvg'] = (
            (df['low'].shift(-2) - df['high']) / df['close'] > min_gap_pct/100
        )

        # Bearish FVG: gap down (candle 1 low > candle 3 high)
        df['bearish_fvg'] = (
            (df['low'] - df['high'].shift(-2)) / df['close'] > min_gap_pct/100
        )

        # Store FVG zones
        df['fvg_high'] = np.where(df['bullish_fvg'], df['low'].shift(-2),
                                  np.where(df['bearish_fvg'], df['low'], np.nan))
        df['fvg_low'] = np.where(df['bullish_fvg'], df['high'],
                                 np.where(df['bearish_fvg'], df['high'].shift(-2), np.nan))

        # Forward fill FVG zones for next 20 candles (extended from 10)
        df['fvg_high_ffill'] = df['fvg_high'].fillna(method='ffill', limit=20)
        df['fvg_low_ffill'] = df['fvg_low'].fillna(method='ffill', limit=20)

        # Check if current price is in FVG zone
        df['in_fvg'] = (
            (df['close'] >= df['fvg_low_ffill']) &
            (df['close'] <= df['fvg_high_ffill'])
        )

        # FVG size feature
        df['fvg_size'] = (df['fvg_high_ffill'] - df['fvg_low_ffill']) / df['close']
        df['fvg_size'] = df['fvg_size'].fillna(0)

        return df

    @staticmethod
    def calculate_market_regime(df, adx_period=14, trend_threshold=20):
        """
        Classify market regime: TRENDING, STRONG_UPTREND, STRONG_DOWNTREND
        ENHANCED: Lowered trend threshold from 25 to 20 for more trending signals
        """
        df = df.copy()

        # Calculate ADX for trend strength
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Directional Movement
        df['high_diff'] = df['high'] - df['high'].shift(1)
        df['low_diff'] = df['low'].shift(1) - df['low']

        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )
        df['minus_dm'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'],
            0
        )

        # Smooth with Wilder's smoothing
        df['atr'] = df['tr'].ewm(alpha=1/adx_period, adjust=False).mean()
        df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/adx_period, adjust=False).mean() / df['atr'])

        # ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].ewm(alpha=1/adx_period, adjust=False).mean()

        # Moving averages for trend direction
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Classify regime
        df['market_regime'] = 'RANGING'

        # Trending markets
        trending_mask = df['adx'] > trend_threshold
        df.loc[trending_mask, 'market_regime'] = 'TRENDING'

        # Strong uptrend
        strong_up_mask = (
            (df['adx'] > trend_threshold) &
            (df['plus_di'] > df['minus_di']) &
            (df['close'] > df['sma_50'])
        )
        df.loc[strong_up_mask, 'market_regime'] = 'STRONG_UPTREND'

        # Strong downtrend
        strong_down_mask = (
            (df['adx'] > trend_threshold) &
            (df['minus_di'] > df['plus_di']) &
            (df['close'] < df['sma_50'])
        )
        df.loc[strong_down_mask, 'market_regime'] = 'STRONG_DOWNTREND'

        # One-hot encode regime
        df['regime_ranging'] = (df['market_regime'] == 'RANGING').astype(int)
        df['regime_trending'] = (df['market_regime'] == 'TRENDING').astype(int)
        df['regime_strong_up'] = (df['market_regime'] == 'STRONG_UPTREND').astype(int)
        df['regime_strong_down'] = (df['market_regime'] == 'STRONG_DOWNTREND').astype(int)

        return df

    @staticmethod
    def detect_overstretched_move(df, lookback=14, threshold=2):
        """
        Detect overstretched moves to avoid entering
        ENHANCED: More nuanced overstretched detection
        """
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=lookback).mean()
        df['bb_std'] = df['close'].rolling(window=lookback).std()
        df['bb_upper'] = df['bb_middle'] + (threshold * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (threshold * df['bb_std'])

        # Price position in BB
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # ATR for volatility
        if 'atr' not in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=lookback).mean()

        # Price distance from MA in ATR units
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['price_distance_atr'] = abs(df['close'] - df['ma_20']) / (df['atr'] + 1e-10)

        # ENHANCED: Only mark as overstretched if EXTREMELY overbought/oversold
        df['overstretched'] = (
            (df['rsi'] > 80) | (df['rsi'] < 20) |  # More extreme RSI levels
            (df['price_distance_atr'] > 3.0)  # Increased threshold
        ).astype(int)

        return df

    @staticmethod
    def calculate_chandelier_exit(df, period=22, multiplier=3):
        """
        Calculate Chandelier Exit for trailing stop loss
        """
        df = df.copy()

        # ATR calculation (reuse if already exists)
        if 'atr' not in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=period).mean()

        # Chandelier Exit levels
        df['highest_high'] = df['high'].rolling(window=period).max()
        df['lowest_low'] = df['low'].rolling(window=period).min()

        df['chandelier_long_exit'] = df['highest_high'] - (multiplier * df['atr'])
        df['chandelier_short_exit'] = df['lowest_low'] + (multiplier * df['atr'])

        return df

    @staticmethod
    def add_time_features(df):
        """
        Add time-based features
        """
        df = df.copy()

        # Hour and minute
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute

        # Day of week (0 = Monday, 4 = Friday)
        df['day_of_week'] = df['datetime'].dt.dayofweek

        # Week of month (1-5)
        df['week_of_month'] = (df['datetime'].dt.day - 1) // 7 + 1

        # Days to month end (proxy for expiry awareness)
        df['day_of_month'] = df['datetime'].dt.day
        df['days_in_month'] = df['datetime'].dt.days_in_month
        df['days_to_month_end'] = df['days_in_month'] - df['day_of_month']

        # Flag last week of month (expiry week for monthly options)
        df['is_expiry_week'] = (df['days_to_month_end'] <= 7).astype(int)

        # Session periods
        df['is_opening_hour'] = (df['hour'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour'] == 15).astype(int)
        df['is_mid_session'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)

        return df

    @staticmethod
    def add_technical_indicators(df):
        """
        Add additional technical indicators
        ENHANCED: Added more momentum and volatility features
        """
        df = df.copy()

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Add ema_12 and ema_26 for MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # Price position relative to MAs
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-10)
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-10)

        # MA crossovers
        df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
        df['ema_20_50_cross'] = (df['ema_20'] > df['ema_50']).astype(int)

        # Momentum
        df['roc_5'] = ((df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)) * 100
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)) * 100
        df['roc_20'] = ((df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-10)) * 100

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Volume features
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)

        # Volume trend
        volume_5 = df['volume'].rolling(window=5).mean()
        volume_20 = df['volume'].rolling(window=20).mean()
        df['volume_trend'] = volume_5 / (volume_20 + 1e-10)

        # Candle features
        df['candle_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['candle_body'] = abs(df['close'] - df['open']) / (df['close'] + 1e-10)
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['close'] + 1e-10)
        df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['close'] + 1e-10)

        # Bullish/bearish candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))

        # Volatility
        df['returns_std_5'] = df['returns'].rolling(window=5).std()
        df['returns_std_20'] = df['returns'].rolling(window=20).std()

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Fill any remaining NaN values in features with forward fill then 0
        feature_cols = [col for col in df.columns if col not in ['datetime', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']]
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(method='ffill').fillna(0)

        return df

    @staticmethod
    def create_target_variable(df, horizon=5):
        """
        Create target variable for ML

        Parameters:
        -----------
        horizon : int
            Number of periods to look ahead for returns
        """
        df = df.copy()

        # Future returns
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # Binary direction (1 = up, 0 = down)
        df['future_direction'] = (df['future_return'] > 0).astype(int)

        # Future high/low for exit analysis
        df['future_high'] = df['high'].rolling(window=horizon).max().shift(-horizon)
        df['future_low'] = df['low'].rolling(window=horizon).min().shift(-horizon)

        return df

    def engineer_all_features(self, df, target_horizon=5):
        """
        Apply all feature engineering steps
        """
        print("\nEngineering features...")

        df = self.calculate_fibonacci_levels(df, lookback=100)
        print("✓ Fibonacci levels calculated")

        df = self.identify_fvg(df, min_gap_pct=0.05)
        print("✓ FVG identified")

        df = self.calculate_market_regime(df, adx_period=14, trend_threshold=20)
        print("✓ Market regime classified")

        df = self.detect_overstretched_move(df, lookback=14, threshold=2)
        print("✓ Overstretched detection added")

        df = self.calculate_chandelier_exit(df, period=22, multiplier=3)
        print("✓ Chandelier exit calculated")

        df = self.add_time_features(df)
        print("✓ Time features added")

        df = self.add_technical_indicators(df)
        print("✓ Technical indicators added")

        df = self.create_target_variable(df, horizon=target_horizon)
        print("✓ Target variable created")

        print(f"\nTotal columns: {len(df.columns)}")

        return df

#=============================================================================
# PART 3: DATASET PREPARATION
#=============================================================================

class DatasetPreparer:
    """Prepare final dataset for ML training"""

    @staticmethod
    def get_feature_columns():
        """
        Define feature columns for ML model
        ENHANCED: Added more features for better predictions
        """
        feature_cols = [
            # Fibonacci & OTE
            'fib_0.500', 'fib_0.600', 'fib_0.700', 'distance_from_ote', 'in_ote_zone',
            'in_premium_zone', 'in_discount_zone',

            # FVG
            'in_fvg', 'bullish_fvg', 'bearish_fvg', 'fvg_size',

            # Market Regime
            'adx', 'plus_di', 'minus_di',
            'regime_ranging', 'regime_trending', 'regime_strong_up', 'regime_strong_down',

            # Overstretched
            'rsi', 'bb_position', 'price_distance_atr', 'overstretched',

            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'price_vs_sma20', 'price_vs_sma50',
            'ema_5_20_cross', 'ema_20_50_cross',

            # Momentum
            'roc_5', 'roc_10', 'roc_20',
            'macd', 'macd_signal', 'macd_hist',

            # Volume
            'volume_ratio', 'volume_trend',

            # Candle features
            'candle_range', 'candle_body', 'upper_wick', 'lower_wick', 'is_bullish',

            # Time features
            'hour', 'day_of_week', 'week_of_month', 'days_to_month_end',
            'is_expiry_week', 'is_opening_hour', 'is_closing_hour', 'is_mid_session',

            # Lags
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5', 'returns_lag_10',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3',

            # Volatility
            'atr', 'bb_std', 'returns_std_5', 'returns_std_20',

            # Chandelier
            'chandelier_long_exit', 'chandelier_short_exit',
        ]

        return feature_cols

    @staticmethod
    def prepare_ml_dataset(df, feature_cols=None):
        """
        Prepare clean dataset for ML
        IMPROVED: Better NaN handling and validation

        Returns:
        --------
        X, y, metadata_df, available_features
        """
        if feature_cols is None:
            feature_cols = DatasetPreparer.get_feature_columns()

        # Target column
        target_col = 'future_direction'

        # Metadata columns to keep
        metadata_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume',
                        'in_ote_zone', 'in_fvg', 'market_regime',
                        'chandelier_long_exit', 'chandelier_short_exit',
                        'future_return', 'future_high', 'future_low',
                        'bullish_fvg', 'bearish_fvg', 'adx', 'rsi', 'overstretched']

        # Check which features actually exist in df
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]

        if missing_features:
            print(f"\nWarning: {len(missing_features)} features not found in dataframe:")
            for mf in missing_features[:15]:
                print(f"  - {mf}")
            if len(missing_features) > 15:
                print(f"  ... and {len(missing_features) - 15} more")

        # Check if we have the target
        if target_col not in df.columns:
            print(f"\nERROR: Target column '{target_col}' not found!")
            return None, None, None, []

        # Select columns
        required_cols = available_features + [target_col] + metadata_cols
        required_cols = list(set(required_cols))  # Remove duplicates

        # Filter to existing columns
        existing_cols = [col for col in required_cols if col in df.columns]
        df_subset = df[existing_cols].copy()

        print(f"\nDataset before cleaning: {len(df_subset):,} rows")

        # Only drop rows where target is NaN
        df_clean = df_subset.dropna(subset=[target_col])
        print(f"After dropping NaN targets: {len(df_clean):,} rows")

        # For features, only drop if ALL features are NaN (keep if at least some are valid)
        # First, fill infinite values
        for col in available_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        # Drop rows where more than 50% of features are NaN
        threshold = len(available_features) * 0.5
        df_clean = df_clean.dropna(subset=available_features, thresh=int(threshold))
        print(f"After dropping rows with >50% NaN features: {len(df_clean):,} rows")

        # Fill remaining NaN in features with 0 (conservative)
        for col in available_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)

        # Separate features, target, and metadata
        X = df_clean[available_features]
        y = df_clean[target_col]
        metadata_df = df_clean[[col for col in metadata_cols if col in df_clean.columns]]

        print(f"\n{'='*80}")
        print("DATASET PREPARATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total features: {len(available_features)}")
        print(f"Total samples: {len(X):,}")

        if len(X) > 0:
            print(f"Date range: {metadata_df['datetime'].min()} to {metadata_df['datetime'].max()}")
            print(f"\nTarget distribution:")
            print(f"  Up moves (1): {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
            print(f"  Down moves (0): {(len(y)-y.sum()):,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

            # Check for OTE entries
            if 'in_ote_zone' in metadata_df.columns:
                ote_count = metadata_df['in_ote_zone'].sum()
                print(f"\nOTE zone entries: {ote_count:,} ({ote_count/len(metadata_df)*100:.2f}%)")
        else:
            print("\nWARNING: No valid samples after preparation!")

        return X, y, metadata_df, available_features

#=============================================================================
# PART 4: MODEL TRAINING
#=============================================================================

class XGBoostTrainer:
    """Train XGBoost model for trading signals"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.best_threshold = 0.5
        self.best_iteration = None

    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler
        """
        print("\n" + "="*80)
        print("SCALING FEATURES")
        print("="*80)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print("✓ Features scaled using StandardScaler")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_class_imbalance(self, y_train):
        """
        Calculate scale_pos_weight for XGBoost
        """
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive

        print(f"\nClass distribution:")
        print(f"  Negative (0): {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
        print(f"  Positive (1): {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
        print(f"  Scale pos weight: {scale_pos_weight:.3f}")

        return scale_pos_weight

    def train_model(self, X_train, y_train, X_val, y_val, scale_pos_weight):
        """
        Train XGBoost model with optimized hyperparameters
        """
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)

        # XGBoost parameters (optimized for trading)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 9,
            'learning_rate': 0.0158169,
            'n_estimators': 400,
            'min_child_weight': 3,
            'subsample': 0.6389274219025062,
            'colsample_bytree': 0.9987349142404867,
            'colsample_bylevel': 0.7705609360826743,
            'gamma': 0.28155316324336543,
            'reg_alpha': 1.7346201315788978,
            'reg_lambda': 3.2706207110759644,
            'max_delta_step': 4,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
        }

        # Create model
        self.model = xgb.XGBClassifier(**params)

        # Train with early stopping
        print("\nTraining in progress...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50
        )

        print("\n✓ Model training complete!")

        # Get best iteration
        if hasattr(self.model, 'best_iteration'):
            print(f"✓ Best iteration: {self.model.best_iteration}")
            self.best_iteration = self.model.best_iteration
        else:
            print(f"✓ Training completed all {params['n_estimators']} iterations")
            self.best_iteration = params['n_estimators']

        return self.model

    def evaluate_model(self, X, y, dataset_name="Dataset"):
        """
        Evaluate model performance
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING ON {dataset_name.upper()}")
        print(f"{'='*80}")

        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > self.best_threshold).astype(int)

        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Down', 'Up']))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # AUC Score
        auc = roc_auc_score(y, y_pred_proba)
        print(f"\nROC-AUC Score: {auc:.4f}")

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc': auc,
            'confusion_matrix': cm
        }

    def find_optimal_threshold(self, X_val, y_val):
        """
        Find optimal probability threshold using validation set
        """
        print("\n" + "="*80)
        print("OPTIMIZING PROBABILITY THRESHOLD")
        print("="*80)

        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        # Calculate precision and recall for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

        # F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Find threshold with best F1
        best_idx = np.argmax(f1_scores[:-1])
        self.best_threshold = thresholds[best_idx]

        print(f"\nOptimal threshold: {self.best_threshold:.4f}")
        print(f"Precision at optimal: {precisions[best_idx]:.4f}")
        print(f"Recall at optimal: {recalls[best_idx]:.4f}")
        print(f"F1 Score at optimal: {f1_scores[best_idx]:.4f}")

        return self.best_threshold

    def plot_feature_importance(self, top_n=30):
        """
        Plot feature importance
        """
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\n" + "="*80)
        print(f"TOP {top_n} IMPORTANT FEATURES")
        print("="*80)
        print(feature_importance_df.head(top_n).to_string(index=False))

        # Plot
        plt.figure(figsize=(12, 10))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/feature_importance.png', dpi=150, bbox_inches='tight')
        print("\n✓ Feature importance plot saved")
        plt.show()

        return feature_importance_df

    def save_model(self):
        """
        Save trained model and scaler
        """
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)

        # Save XGBoost model
        self.model.save_model('/content/drive/MyDrive/xgboost_model.json')
        print("✓ XGBoost model saved")

        # Save scaler
        with open('/content/drive/MyDrive/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Scaler saved")

        # Save threshold
        with open('/content/drive/MyDrive/best_threshold.pkl', 'wb') as f:
            pickle.dump(self.best_threshold, f)
        print(f"✓ Best threshold saved: {self.best_threshold:.4f}")

        # Save feature names
        with open('/content/drive/MyDrive/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print("✓ Feature names saved")

#=============================================================================
# PART 5: BACKTESTING ENGINE
#=============================================================================

class TradingBacktester:
    """
    Backtest trading strategy with realistic execution
    ENHANCED: Relaxed filters for more trading opportunities
    """

    def __init__(self, initial_capital=100000, position_size_pct=0.10,
                 commission_pct=0.0003, slippage_pct=0.0001):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_names = None

    def load_model(self):
        """Load trained model and preprocessing objects"""
        print("\n" + "="*80)
        print("LOADING TRAINED MODEL")
        print("="*80)

        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model('/content/drive/MyDrive/xgboost_model.json')
        print("✓ XGBoost model loaded")

        # Load scaler
        with open('/content/drive/MyDrive/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded")

        # Load threshold
        with open('/content/drive/MyDrive/best_threshold.pkl', 'rb') as f:
            self.threshold = pickle.load(f)
        print(f"✓ Threshold loaded: {self.threshold:.4f}")

        # Load feature names
        with open('/content/drive/MyDrive/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        print(f"✓ Feature names loaded: {len(self.feature_names)} features")

    def generate_signals(self, X, metadata_df):
        """
        Generate trading signals using trained model
        ENHANCED: Relaxed confluence filters for more signals

        Returns:
        --------
        signals : array
            1 = Long, 0 = No trade
        probabilities : array
            Model prediction probabilities
        """
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        signals = (probabilities > self.threshold).astype(int)

        # ENHANCED FILTERING LOGIC
        # Filter 1: OTE zone OR FVG (changed from AND to OR for more signals)
        confluence_filter = metadata_df['in_ote_zone'] | metadata_df['in_fvg']

        # Filter 2: Trending markets OR bullish FVG (relaxed regime requirement)
        regime_filter = (
            metadata_df['market_regime'].isin(['TRENDING', 'STRONG_UPTREND', 'STRONG_DOWNTREND']) |
            metadata_df['bullish_fvg']
        )

        # Filter 3: Not extremely overstretched (keep moderate overstretched)
        not_overstretched = ~metadata_df['overstretched']

        # Filter 4: Minimum ADX for some directional movement
        min_adx = metadata_df['adx'] > 15  # Reduced from 20

        # Combined filter (more lenient)
        final_filter = confluence_filter & (regime_filter | not_overstretched) & min_adx

        # Apply filter to signals
        filtered_signals = signals.copy()
        filtered_signals[~final_filter] = 0

        print(f"\nSignal Generation:")
        print(f"  Raw ML signals: {signals.sum():,}")
        print(f"  After OTE/FVG filter: {(signals & confluence_filter).sum():,}")
        print(f"  After regime filter: {(signals & confluence_filter & regime_filter).sum():,}")
        print(f"  After overstretched filter: {(signals & confluence_filter & not_overstretched).sum():,}")
        print(f"  After ADX filter: {(signals & final_filter).sum():,}")
        print(f"  Final signals: {filtered_signals.sum():,} ({filtered_signals.sum()/len(filtered_signals)*100:.2f}%)")

        return filtered_signals, probabilities

    def backtest(self, X, metadata_df, signals, probabilities):
        """
        Run backtest with Chandelier exit strategy
        ENHANCED: Added take-profit target

        Returns:
        --------
        trades_df : DataFrame
            Detailed trade log
        equity_curve : array
            Portfolio value over time
        """
        print("\n" + "="*80)
        print("RUNNING BACKTEST")
        print("="*80)

        # Initialize
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []

        in_position = False
        entry_price = 0
        entry_idx = 0
        entry_capital = 0
        shares = 0
        stop_loss = 0
        take_profit = 0

        # Iterate through each bar
        for i in range(len(metadata_df)):
            current_price = metadata_df.iloc[i]['close']
            current_time = metadata_df.iloc[i]['datetime']

            # ENTRY LOGIC
            if not in_position and signals[i] == 1:
                # Calculate position size
                position_value = capital * self.position_size_pct

                # Account for slippage on entry
                entry_price = current_price * (1 + self.slippage_pct)

                # Calculate shares (for index, use points as "shares")
                shares = position_value / entry_price

                # Commission on entry
                commission = position_value * self.commission_pct

                # Update capital
                entry_capital = capital
                capital -= commission

                # Set stop loss using Chandelier exit
                stop_loss = metadata_df.iloc[i]['chandelier_long_exit']

                # ENHANCED: Set take profit at 2:1 risk-reward
                risk = entry_price - stop_loss
                take_profit = entry_price + (2 * risk)

                # Enter position
                in_position = True
                entry_idx = i

            # EXIT LOGIC
            elif in_position:
                # Update trailing stop (Chandelier)
                new_stop = metadata_df.iloc[i]['chandelier_long_exit']
                stop_loss = max(stop_loss, new_stop)  # Trail up only

                # Check exit conditions
                exit_triggered = False
                exit_reason = ""

                # 1. Take profit hit
                if current_price >= take_profit:
                    exit_triggered = True
                    exit_reason = "Take Profit"
                    exit_price = take_profit * (1 - self.slippage_pct)

                # 2. Stop loss hit
                elif current_price <= stop_loss:
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                    exit_price = stop_loss * (1 - self.slippage_pct)

                # 3. Opposite signal (low probability)
                elif signals[i] == 0 and probabilities[i] < 0.3:
                    exit_triggered = True
                    exit_reason = "Opposite Signal"
                    exit_price = current_price * (1 - self.slippage_pct)

                # 4. End of data
                elif i == len(metadata_df) - 1:
                    exit_triggered = True
                    exit_reason = "End of Data"
                    exit_price = current_price * (1 - self.slippage_pct)

                # Execute exit
                if exit_triggered:
                    # Calculate P&L
                    position_value = shares * exit_price
                    commission = position_value * self.commission_pct
                    pnl = position_value - (shares * entry_price) - commission
                    pnl_pct = (pnl / entry_capital) * 100

                    # Update capital
                    capital += pnl

                    # Record trade
                    trade = {
                        'entry_time': metadata_df.iloc[entry_idx]['datetime'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'return': (exit_price / entry_price) - 1,
                        'exit_reason': exit_reason,
                        'holding_period': i - entry_idx,
                        'entry_capital': entry_capital,
                        'exit_capital': capital,
                        'in_ote_zone': metadata_df.iloc[entry_idx]['in_ote_zone'],
                        'in_fvg': metadata_df.iloc[entry_idx]['in_fvg'],
                        'market_regime': metadata_df.iloc[entry_idx]['market_regime'],
                        'probability': probabilities[entry_idx],
                    }
                    trades.append(trade)

                    # Reset position
                    in_position = False
                    entry_price = 0
                    entry_idx = 0
                    shares = 0
                    stop_loss = 0
                    take_profit = 0

            # Update equity curve
            if in_position:
                # Mark-to-market
                current_position_value = shares * current_price
                current_equity = capital + current_position_value - (shares * entry_price)
            else:
                current_equity = capital

            equity_curve.append(current_equity)

        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)

        if len(trades_df) > 0:
            trades_df['profitable'] = (trades_df['pnl'] > 0).astype(int)

        print(f"\n✓ Backtest complete!")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Final capital: ${capital:,.2f}")
        print(f"  Total return: {((capital - self.initial_capital) / self.initial_capital * 100):.2f}%")

        return trades_df, np.array(equity_curve)

#=============================================================================
# PART 6: PERFORMANCE METRICS
#=============================================================================

class PerformanceEvaluator:
    """Calculate comprehensive trading performance metrics"""

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        return sharpe

    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.0):
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        mean_return = returns.mean()
        downside_std = downside_returns.std()
        sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
        return sortino

    @staticmethod
    def calculate_max_drawdown(equity_curve):
        if len(equity_curve) < 2:
            return 0.0
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        return max_dd

    @staticmethod
    def calculate_trade_metrics(trades_df):
        if len(trades_df) == 0:
            return {}

        win_rate = (trades_df['profitable'].sum() / len(trades_df)) * 100
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        win_pct = trades_df['profitable'].mean()
        loss_pct = 1 - win_pct
        expectancy = (win_pct * avg_win) - (loss_pct * abs(avg_loss))

        avg_holding_period = trades_df['holding_period'].mean()

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'expectancy': expectancy,
            'avg_holding_period': avg_holding_period,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    @staticmethod
    def generate_performance_report(trades_df, equity_curve, initial_capital):
        print("\n" + "="*80)
        print("PERFORMANCE REPORT")
        print("="*80)

        if len(trades_df) == 0:
            print("\nNo trades executed!")
            return {}

        returns = trades_df['return'].values
        sharpe = PerformanceEvaluator.calculate_sharpe_ratio(returns)
        sortino = PerformanceEvaluator.calculate_sortino_ratio(returns)
        max_dd = PerformanceEvaluator.calculate_max_drawdown(equity_curve)
        trade_metrics = PerformanceEvaluator.calculate_trade_metrics(trades_df)
        total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100

        print("\n📊 RISK-ADJUSTED RETURNS")
        print("-" * 80)
        print(f"Sharpe Ratio:      {sharpe:>10.4f}")
        print(f"Sortino Ratio:     {sortino:>10.4f}")
        print(f"Max Drawdown:      {max_dd*100:>10.2f}%")

        print("\n💰 TRADE STATISTICS")
        print("-" * 80)
        print(f"Total Trades:      {trade_metrics['total_trades']:>10,}")
        print(f"Win Rate:          {trade_metrics['win_rate']:>10.2f}%")
        print(f"Profit Factor:     {trade_metrics['profit_factor']:>10.2f}")
        print(f"Expectancy:        ${trade_metrics['expectancy']:>10,.2f}")

        print("\n📈 RETURNS")
        print("-" * 80)
        print(f"Total Return:      {total_return:>10.2f}%")
        print(f"Gross Profit:      ${trade_metrics['gross_profit']:>10,.2f}")
        print(f"Gross Loss:        ${trade_metrics['gross_loss']:>10,.2f}")

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'total_return_pct': total_return,
            **trade_metrics
        }

#=============================================================================
# PART 7: VISUALIZATION
#=============================================================================

def plot_backtest_results(equity_curve, trades_df, metadata_df, dataset_name="Test"):
    """Plot comprehensive backtest visualizations"""
    fig = plt.figure(figsize=(20, 12))

    # 1. Equity Curve
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(equity_curve, linewidth=2, color='green')
    ax1.set_title(f'{dataset_name} - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (bars)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=equity_curve[0], color='red', linestyle='--', linewidth=1)

    # 2. Drawdown
    ax2 = plt.subplot(3, 2, 2)
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='darkred', linewidth=1)
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (bars)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    if len(trades_df) > 0:
        # 3. Trade P&L Distribution
        ax3 = plt.subplot(3, 2, 3)
        ax3.hist(trades_df['pnl'], bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative Returns
        ax4 = plt.subplot(3, 2, 4)
        cumulative_returns = (trades_df['pnl'].cumsum() / equity_curve[0]) * 100
        ax4.plot(cumulative_returns.values, linewidth=2, color='blue')
        ax4.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Exit Reasons
        ax5 = plt.subplot(3, 2, 5)
        exit_counts = trades_df['exit_reason'].value_counts()
        ax5.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%')
        ax5.set_title('Exit Reasons', fontsize=14, fontweight='bold')

        # 6. Returns over Time
        ax6 = plt.subplot(3, 2, 6)
        trades_df['trade_num'] = range(len(trades_df))
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
        ax6.bar(trades_df['trade_num'], trades_df['pnl'], color=colors, alpha=0.6)
        ax6.set_title('Individual Trade Returns', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Trade Number')
        ax6.set_ylabel('P&L ($)')
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/{dataset_name.lower()}_backtest_results.png',
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Backtest visualization saved")
    plt.show()

#=============================================================================
# MAIN EXECUTION
#=============================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("FINNIFTY COMPLETE ALGORITHMIC TRADING SYSTEM")
    print("Enhanced Version 2.0 - Better Signal Generation")
    print("="*80)

    # =========================================================================
    # STEP 1: DATA LOADING AND FEATURE ENGINEERING
    # =========================================================================

    loader = DataLoader()
    engineer = FeatureEngineer()
    preparer = DatasetPreparer()

    # Load and split data
    train_df, val_df, test_df = loader.split_train_val_test()

    # Engineer features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - TRAINING SET")
    print("="*80)
    train_df = engineer.engineer_all_features(train_df, target_horizon=5)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING - VALIDATION SET")
    print("="*80)
    val_df = engineer.engineer_all_features(val_df, target_horizon=5)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING - TEST SET")
    print("="*80)
    test_df = engineer.engineer_all_features(test_df, target_horizon=5)

    # Prepare datasets
    print("\n" + "="*80)
    print("PREPARING DATASETS")
    print("="*80)

    X_train, y_train, train_metadata, feature_cols = preparer.prepare_ml_dataset(train_df)
    X_val, y_val, val_metadata, _ = preparer.prepare_ml_dataset(val_df, feature_cols)
    X_test, y_test, test_metadata, _ = preparer.prepare_ml_dataset(test_df, feature_cols)

    # Safety check: ensure we have data
    if len(X_train) == 0:
        print("\n" + "="*80)
        print("ERROR: Training set is empty!")
        print("="*80)
        raise ValueError("Training set has 0 samples. Check data preprocessing.")

    if len(X_val) == 0:
        print("\n" + "="*80)
        print("WARNING: Validation set is empty! Using a portion of training data.")
        print("="*80)
        # Split training data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        val_metadata = train_metadata.iloc[:len(X_val)].copy()
        print(f"Created validation set: {len(X_val):,} samples")

    if len(X_test) == 0:
        print("\n" + "="*80)
        print("WARNING: Test set is empty! Using a portion of training data.")
        print("="*80)
        # Use validation set as test set
        X_test, y_test = X_val, y_val
        test_metadata = val_metadata.copy()
        print(f"Using validation as test set: {len(X_test):,} samples")

    print(f"\nFinal dataset sizes:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")

    # =========================================================================
    # STEP 2: MODEL TRAINING
    # =========================================================================

    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)

    trainer = XGBoostTrainer()
    trainer.feature_names = feature_cols

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_val, X_test
    )

    # Handle class imbalance
    scale_pos_weight = trainer.handle_class_imbalance(y_train)

    # Train model
    model = trainer.train_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        scale_pos_weight
    )

    # Find optimal threshold
    best_threshold = trainer.find_optimal_threshold(X_val_scaled, y_val)

    # Evaluate
    val_results = trainer.evaluate_model(X_val_scaled, y_val, "VALIDATION SET")
    test_results = trainer.evaluate_model(X_test_scaled, y_test, "TEST SET")

    # Plot feature importance
    feature_importance_df = trainer.plot_feature_importance(top_n=30)

    # Save model
    trainer.save_model()

    # =========================================================================
    # STEP 3: BACKTESTING
    # =========================================================================

    print("\n" + "="*80)
    print("BACKTESTING")
    print("="*80)

    backtester = TradingBacktester(
        initial_capital=100000,
        position_size_pct=0.10,
        commission_pct=0.0003,
        slippage_pct=0.0001
    )

    # Load model for backtesting
    backtester.model = model
    backtester.scaler = trainer.scaler
    backtester.threshold = trainer.best_threshold
    backtester.feature_names = feature_cols

    # Generate signals
    signals, probabilities = backtester.generate_signals(X_test, test_metadata)

    # Run backtest
    trades_df, equity_curve = backtester.backtest(X_test, test_metadata, signals, probabilities)

    # Calculate metrics
    evaluator = PerformanceEvaluator()
    metrics = evaluator.generate_performance_report(
        trades_df, equity_curve, backtester.initial_capital
    )

    # Save results
    if len(trades_df) > 0:
        trades_df.to_csv('/content/drive/MyDrive/trade_log.csv', index=False)
        print("\n✓ Trade log saved")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('/content/drive/MyDrive/backtest_metrics.csv', index=False)
    print("✓ Metrics saved")

    # Plot results
    plot_backtest_results(equity_curve, trades_df, test_metadata, "Test Set")

    print("\n" + "="*80)
    print("SYSTEM COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - xgboost_model.json")
    print("  - scaler.pkl")
    print("  - best_threshold.pkl")
    print("  - feature_names.pkl")
    print("  - feature_importance.png")
    print("  - trade_log.csv")
    print("  - backtest_metrics.csv")
    print("  - test_set_backtest_results.png")