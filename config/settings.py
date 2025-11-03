import os
from dotenv import load_dotenv

load_dotenv()

# Binance Configuration
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'testnet': True  # Start with testnet
}

# Reddit Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT')
}

# App Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
SUBREDDITS = ['CryptoCurrency', 'binance', 'CryptoMarkets']
UPDATE_INTERVAL = 60  # seconds
SENTIMENT_THRESHOLD = 0.1  # Minimum sentiment score to consider