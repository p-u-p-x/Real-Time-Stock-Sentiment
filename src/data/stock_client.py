import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    def __init__(self):
        self.price_data = {}

    def get_historical_data(self, symbol, interval='1d', lookback_days=7):
        """Get historical stock data using yfinance - SIMPLIFIED VERSION"""
        try:
            logger.info(f"Fetching historical data for {symbol}")

            # Use yfinance to get data
            stock = yf.Ticker(symbol)

            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Download historical data
            hist_data = stock.history(start=start_date, end=end_date, interval=interval)

            if hist_data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Reset index to get Date as a column
            hist_data = hist_data.reset_index()

            # Create our standard format DataFrame
            df = pd.DataFrame({
                'timestamp': hist_data['Date'],
                'open': hist_data['Open'].astype(float),
                'high': hist_data['High'].astype(float),
                'low': hist_data['Low'].astype(float),
                'close': hist_data['Close'].astype(float),
                'volume': hist_data['Volume'].astype(float)
            })

            # Calculate basic technical indicators
            df = self.calculate_basic_indicators(df)
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_basic_indicators(self, df):
        """Calculate basic technical indicators for stocks"""
        try:
            if len(df) == 0:
                return df

            # Simple moving average (5-day)
            if len(df) >= 5:
                df['sma_5'] = df['close'].rolling(window=5).mean()
            else:
                df['sma_5'] = df['close']

            # RSI calculation
            if len(df) > 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            else:
                df['rsi'] = 50  # Default value

            # Price change percentage
            df['price_change'] = df['close'].pct_change() * 100

            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def get_current_price(self, symbol):
        """Get current stock price"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('currentPrice',
                                     info.get('regularMarketPrice',
                                              info.get('previousClose', 0)))
            return float(current_price)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0


# Alternative ultra-simple method
def get_stock_data_ultra_simple(symbol, days=7):
    """Ultra-simple method to get stock data"""
    try:
        print(f"🔄 Fetching {symbol} data...")
        stock = yf.Ticker(symbol)

        # Get the last 'days' days of data
        hist = stock.history(period=f"{days}d")

        if hist.empty:
            print(f"❌ No data for {symbol}")
            return pd.DataFrame()

        # Create simple dataframe
        df = pd.DataFrame()
        df['timestamp'] = hist.index
        df['open'] = hist['Open'].values
        df['high'] = hist['High'].values
        df['low'] = hist['Low'].values
        df['close'] = hist['Close'].values
        df['volume'] = hist['Volume'].values

        # Simple RSI
        if len(df) > 1:
            price_diff = df['close'].diff()
            gain = price_diff.where(price_diff > 0, 0)
            loss = -price_diff.where(price_diff < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = 50

        df = df.fillna(50)
        print(f"✅ Got {len(df)} records for {symbol}")
        return df

    except Exception as e:
        print(f"❌ Error getting {symbol}: {str(e)}")
        return pd.DataFrame()


# Test function
def test_stock_connection():
    print("🧪 Testing Stock Connection...")

    # Test ultra-simple method
    symbols = ['AAPL', 'TSLA', 'MSFT']

    for symbol in symbols:
        print(f"\nTesting {symbol}:")
        data = get_stock_data_ultra_simple(symbol, 3)
        if not data.empty:
            latest = data.iloc[-1]
            print(f"✅ Price: ${latest['close']:.2f}, RSI: {latest['rsi']:.1f}")
        else:
            print(f"❌ Failed")

    return True


if __name__ == "__main__":
    test_stock_connection()