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

# News API Configuration
NEWS_API_CONFIG = {
    'api_key': os.getenv('NEWS_API_KEY'),
    'provider': 'newsapi'
}

# Asset Configuration
CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
STOCK_SYMBOLS = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']

# All symbols combined
ALL_SYMBOLS = CRYPTO_SYMBOLS + STOCK_SYMBOLS

# Social media sources for sentiment
SUBREDDITS = ['stocks', 'investing', 'wallstreetbets', 'CryptoCurrency', 'binance', 'CryptoMarkets', 'technology']

# News sources
NEWS_SOURCES = ['bloomberg', 'reuters', 'financial-times', 'cnbc', 'the-wall-street-journal']
CRYPTO_KEYWORDS = ['bitcoin', 'ethereum', 'crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft']
STOCK_KEYWORDS = ['stocks', 'trading', 'investing', 'market', 'earnings', 'financial']

# App Configuration
UPDATE_INTERVAL = 60  # seconds
SENTIMENT_THRESHOLD = 0.1

# Asset display names
ASSET_DISPLAY_NAMES = {
    # Crypto
    'BTCUSDT': 'Bitcoin', 'ETHUSDT': 'Ethereum', 'ADAUSDT': 'Cardano',
    'DOTUSDT': 'Polkadot', 'LINKUSDT': 'Chainlink', 'SOLUSDT': 'Solana',
    'XRPUSDT': 'Ripple', 'DOGEUSDT': 'Dogecoin',
    # Stocks
    'AAPL': 'Apple Inc.', 'TSLA': 'Tesla Inc.', 'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.', 'GOOGL': 'Alphabet Inc.', 'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms Inc.', 'JPM': 'JPMorgan Chase & Co.',
    'NFLX': 'Netflix Inc.', 'AMD': 'Advanced Micro Devices Inc.'
}