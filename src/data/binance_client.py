import pandas as pd
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.client.API_URL = 'https://testnet.binance.vision/api'  # Use testnet first
        self.price_data = {}

    def get_historical_data(self, symbol, interval='1h', lookback_days=7):
        """Get historical klines data for model training"""
        try:
            logger.info(f"Fetching historical data for {symbol}")
            klines = self.client.get_historical_klines(
                symbol, interval, f"{lookback_days} days ago UTC"
            )

            if not klines:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert to proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate additional features
            df = self.calculate_technical_indicators(df)
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the dataframe"""
        try:
            # Simple moving average
            df['sma_20'] = df['close'].rolling(window=20).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()

            return df.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def get_current_prices(self, symbols):
        """Get current prices for multiple symbols"""
        try:
            prices = {}
            for symbol in symbols:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
            return prices
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}


# Test function
def test_binance_connection():
    from config.settings import BINANCE_CONFIG

    collector = BinanceDataCollector(
        BINANCE_CONFIG['api_key'],
        BINANCE_CONFIG['api_secret']
    )

    # Test historical data
    btc_data = collector.get_historical_data('BTCUSDT', '1h', 1)
    print(f"Fetched {len(btc_data)} BTC records")

    # Test current price
    prices = collector.get_current_prices(['BTCUSDT', 'ETHUSDT'])
    print("Current prices:", prices)

    return collector


if __name__ == "__main__":
    test_binance_connection()