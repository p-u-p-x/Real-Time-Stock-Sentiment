import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    def clean_data(self, df):
        """Clean and prepare data for feature engineering"""
        try:
            df = df.copy()

            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove infinite values and replace with NaN
            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values using forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Remove any remaining NaN rows
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df

    def safe_technical_indicator(self, func, *args, **kwargs):
        """Safely calculate technical indicators with error handling"""
        try:
            result = func(*args, **kwargs)
            if hasattr(result, '__array__'):
                result = np.array(result)
                # Replace infinite values
                result = np.where(np.isfinite(result), result, np.nan)
            return result
        except Exception as e:
            logger.warning(f"Error calculating indicator {func.__name__}: {e}")
            return np.nan

    def create_advanced_ta_features(self, df):
        """Create advanced technical analysis features with robust error handling"""
        try:
            # Clean data first
            df = self.clean_data(df)

            if df.empty:
                logger.error("No data after cleaning")
                return df

            # === MOMENTUM INDICATORS ===
            # RSI with multiple timeframes
            df['rsi_14'] = self.safe_technical_indicator(RSIIndicator(df['close'], window=14).rsi)
            df['rsi_21'] = self.safe_technical_indicator(RSIIndicator(df['close'], window=21).rsi)
            df['rsi_7'] = self.safe_technical_indicator(RSIIndicator(df['close'], window=7).rsi)

            # Stochastic Oscillator
            stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14)
            df['stoch_k'] = self.safe_technical_indicator(stoch.stoch)
            df['stoch_d'] = self.safe_technical_indicator(stoch.stoch_signal)

            # Williams %R
            williams = WilliamsRIndicator(df['high'], df['low'], df['close'])
            df['williams_r'] = self.safe_technical_indicator(williams.williams_r)

            # === TREND INDICATORS ===
            # MACD
            macd = MACD(df['close'])
            df['macd'] = self.safe_technical_indicator(macd.macd)
            df['macd_signal'] = self.safe_technical_indicator(macd.macd_signal)
            df['macd_diff'] = self.safe_technical_indicator(macd.macd_diff)

            # ADX (Trend Strength)
            adx = ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = self.safe_technical_indicator(adx.adx)
            df['adx_pos'] = self.safe_technical_indicator(adx.adx_pos)
            df['adx_neg'] = self.safe_technical_indicator(adx.adx_neg)

            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            df['bb_upper'] = self.safe_technical_indicator(bb.bollinger_hband)
            df['bb_lower'] = self.safe_technical_indicator(bb.bollinger_lband)
            df['bb_middle'] = self.safe_technical_indicator(bb.bollinger_mavg)

            # Calculate derived Bollinger Band features safely
            df['bb_width'] = np.where(
                (df['bb_middle'] > 0) & (df['bb_upper'] > df['bb_lower']),
                (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
                0
            )
            df['bb_position'] = np.where(
                (df['bb_upper'] > df['bb_lower']),
                (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']),
                0.5
            )

            # Average True Range
            atr = AverageTrueRange(df['high'], df['low'], df['close'])
            df['atr'] = self.safe_technical_indicator(atr.average_true_range)

            # === VOLUME INDICATORS ===
            # Volume Weighted Average Price
            vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
            df['vwap'] = self.safe_technical_indicator(vwap.volume_weighted_average_price)

            # On Balance Volume
            obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['obv'] = self.safe_technical_indicator(obv.on_balance_volume)

            # Volume SMA ratios
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = np.where(
                df['volume_sma_20'] > 0,
                df['volume'] / df['volume_sma_20'],
                1
            )

            # === PRICE ACTION FEATURES ===
            # Price changes with bounds
            for period in [1, 2, 3, 4, 6, 8, 12, 24]:
                pct_change = df['close'].pct_change(period)
                # Cap extreme values to ±50%
                df[f'price_change_{period}h'] = np.clip(pct_change, -0.5, 0.5)

            # Rolling price statistics
            df['price_high_24h'] = df['high'].rolling(24).max()
            df['price_low_24h'] = df['low'].rolling(24).min()
            df['price_range_24h'] = np.where(
                df['price_low_24h'] > 0,
                (df['price_high_24h'] - df['price_low_24h']) / df['price_low_24h'],
                0
            )

            # Moving averages and ratios
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_ratio_{period}'] = np.where(
                    df[f'sma_{period}'] > 0,
                    df['close'] / df[f'sma_{period}'],
                    1
                )

            # Price position in recent range (bounded 0-1)
            for period in [6, 12]:
                rolling_min = df['low'].rolling(period).min()
                rolling_max = df['high'].rolling(period).max()
                price_position = np.where(
                    rolling_max > rolling_min,
                    (df['close'] - rolling_min) / (rolling_max - rolling_min),
                    0.5
                )
                df[f'price_position_{period}h'] = np.clip(price_position, 0, 1)

            # === VOLATILITY FEATURES ===
            volatility_6h = df['price_change_1h'].rolling(6).std()
            volatility_12h = df['price_change_1h'].rolling(12).std()
            volatility_24h = df['price_change_1h'].rolling(24).std()

            # Cap volatility to reasonable values
            df['volatility_6h'] = np.clip(volatility_6h, 0, 0.2)
            df['volatility_12h'] = np.clip(volatility_12h, 0, 0.2)
            df['volatility_24h'] = np.clip(volatility_24h, 0, 0.2)

            # === MOMENTUM COMBINATIONS ===
            df['rsi_stoch_combo'] = np.clip((df['rsi_14'] / 100) * (df['stoch_k'] / 100), 0, 1)
            df['macd_momentum'] = np.clip(df['macd'] - df['macd_signal'], -10, 10)

            # === TREND STRENGTH ===
            df['trend_strength'] = np.clip(df['adx'] / 100, 0, 1)

            # Target variable: Next hour price movement (1 = up, 0 = down)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

            # Final data cleaning
            df = self.clean_data(df)

            # Drop columns that might still have issues
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.dropna()

            # Define feature columns (exclude non-numeric and target columns)
            exclude_cols = ['timestamp', 'close_time', 'ignore', 'target', 'symbol']
            self.feature_columns = [col for col in df.columns if
                                    col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

            logger.info(f"Created {len(self.feature_columns)} advanced features")
            return df

        except Exception as e:
            logger.error(f"Error creating advanced features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df

    def create_sentiment_features(self, price_df, sentiment_df):
        """Enhanced sentiment feature engineering"""
        try:
            if sentiment_df.empty:
                return price_df

            # Ensure timestamp is datetime
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])

            # Create a copy to avoid modifying original
            merged_df = price_df.copy()

            # Get unique symbols from sentiment data
            sentiment_symbols = sentiment_df['symbol'].unique()

            # Enhanced sentiment features
            for symbol in sentiment_symbols:
                symbol_sentiment = sentiment_df[sentiment_df['symbol'] == symbol].copy()

                # Sort by timestamp and remove duplicates
                symbol_sentiment = symbol_sentiment.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

                # Create multiple sentiment features
                sentiment_cols = {
                    f"{symbol}_sentiment": 'avg_sentiment',
                    f"{symbol}_mentions": 'total_mentions',
                    f"{symbol}_sentiment_momentum": 'avg_sentiment',
                    f"{symbol}_mention_intensity": 'total_mentions'
                }

                # Initialize columns
                for col in sentiment_cols.keys():
                    merged_df[col] = 0.0

                # Enhanced sentiment merging with time decay
                for idx, row in merged_df.iterrows():
                    ts = row['timestamp']

                    # Find sentiment data within 12 hours with exponential decay
                    time_diff = abs((symbol_sentiment['timestamp'] - ts).dt.total_seconds() / 3600)
                    recent_sentiment = symbol_sentiment[time_diff <= 12]

                    if not recent_sentiment.empty:
                        # Apply time decay (more recent = higher weight)
                        weights = np.exp(-time_diff[recent_sentiment.index] / 6)  # 6-hour half-life
                        total_weight = weights.sum()

                        if total_weight > 0:
                            # Weighted average for sentiment
                            weighted_sentiment = (recent_sentiment['avg_sentiment'] * weights).sum() / total_weight
                            weighted_mentions = (recent_sentiment['total_mentions'] * weights).sum() / total_weight

                            merged_df.at[idx, f"{symbol}_sentiment"] = np.clip(weighted_sentiment, -1, 1)
                            merged_df.at[idx, f"{symbol}_mentions"] = np.clip(weighted_mentions, 0, 1000)

            # Calculate sentiment momentum (rate of change)
            for symbol in sentiment_symbols:
                sentiment_col = f"{symbol}_sentiment"
                mentions_col = f"{symbol}_mentions"

                if sentiment_col in merged_df.columns:
                    merged_df[f"{symbol}_sentiment_momentum"] = np.clip(
                        merged_df[sentiment_col].pct_change(4).fillna(0), -1, 1
                    )

                if mentions_col in merged_df.columns:
                    rolling_avg = merged_df[mentions_col].rolling(12, min_periods=1).mean()
                    merged_df[f"{symbol}_mention_intensity"] = np.where(
                        rolling_avg > 0,
                        merged_df[mentions_col] / rolling_avg,
                        1
                    )
                    merged_df[f"{symbol}_mention_intensity"] = np.clip(
                        merged_df[f"{symbol}_mention_intensity"], 0, 10
                    )

            # Fill NaN values
            sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col or 'mentions' in col]
            merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

            logger.info(f"Added enhanced sentiment features for {len(sentiment_symbols)} symbols")
            return merged_df

        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return price_df

    def prepare_training_data(self, price_data, sentiment_data):
        """Prepare final training dataset with robust error handling"""
        try:
            # Create advanced features
            df_with_features = self.create_advanced_ta_features(price_data)

            if df_with_features.empty:
                logger.error("No features created from price data")
                return pd.DataFrame(), pd.Series(), []

            # Add enhanced sentiment features
            df_with_sentiment = self.create_sentiment_features(df_with_features, sentiment_data)

            # Final data cleaning
            df_with_sentiment = self.clean_data(df_with_sentiment)

            if df_with_sentiment.empty:
                logger.error("No data after final cleaning")
                return pd.DataFrame(), pd.Series(), []

            # Select only numeric feature columns
            feature_cols = [col for col in self.feature_columns if col in df_with_sentiment.columns]

            # Add sentiment columns
            sentiment_cols = [col for col in df_with_sentiment.columns if 'sentiment' in col or 'mentions' in col]
            feature_cols.extend(sentiment_cols)

            # Ensure all selected columns are numeric and finite
            numeric_feature_cols = []
            for col in feature_cols:
                if (col in df_with_sentiment.columns and
                        pd.api.types.is_numeric_dtype(df_with_sentiment[col])):

                    # Check for finite values
                    if not np.isfinite(df_with_sentiment[col]).all():
                        logger.warning(f"Column {col} contains non-finite values, filling with 0")
                        df_with_sentiment[col] = df_with_sentiment[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                    numeric_feature_cols.append(col)
                else:
                    logger.warning(f"Excluding non-numeric column: {col}")

            X = df_with_sentiment[numeric_feature_cols]
            y = df_with_sentiment['target']

            # Final validation
            if X.empty or y.empty:
                logger.error("Empty features or target after processing")
                return pd.DataFrame(), pd.Series(), []

            # Ensure no infinite values remain
            X = X.replace([np.inf, -np.inf], 0)

            logger.info(f"Final training data - X: {X.shape}, y: {y.shape}")
            logger.info(f"Feature columns: {len(numeric_feature_cols)}")

            return X, y, numeric_feature_cols

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), pd.Series(), []