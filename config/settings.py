import os
from dotenv import load_dotenv

load_dotenv()

# Binance Configuration
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'testnet': True
}

# Reddit Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT')
}

# Asset Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']  # Current crypto symbols

# For future stock integration, you could add:
STOCK_SYMBOLS = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

# Social media sources for sentiment
SUBREDDITS = ['stocks', 'investing', 'wallstreetbets', 'CryptoCurrency', 'binance', 'CryptoMarkets']

# App Configuration
UPDATE_INTERVAL = 60  # seconds
SENTIMENT_THRESHOLD = 0.1