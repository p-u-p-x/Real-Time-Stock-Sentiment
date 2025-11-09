import os
from dotenv import load_dotenv

load_dotenv()

# Deployment-safe configuration with fallbacks
def get_env_var(var_name, default=None):
    """Safely get environment variable with fallback"""
    value = os.getenv(var_name, default)
    if value == f"your_{var_name.lower()}_here":
        return None
    return value

# Binance Configuration (for backward compatibility)
BINANCE_CONFIG = {
    'api_key': get_env_var('BINANCE_API_KEY'),
    'api_secret': get_env_var('BINANCE_SECRET_KEY'),
    'testnet': True
}

# Reddit Configuration (for backward compatibility)
REDDIT_CONFIG = {
    'client_id': get_env_var('REDDIT_CLIENT_ID'),
    'client_secret': get_env_var('REDDIT_CLIENT_SECRET'),
    'user_agent': get_env_var('REDDIT_USER_AGENT', 'QuantumTraderAI/1.0')
}

# News API Configuration (for backward compatibility)
NEWS_API_CONFIG = {
    'api_key': get_env_var('NEWS_API_KEY'),
    'provider': 'newsapi'
}

# Individual API Keys (new structure)
REDDIT_CLIENT_ID = get_env_var('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = get_env_var('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = get_env_var('REDDIT_USER_AGENT', 'QuantumTraderAI/1.0')
BINANCE_API_KEY = get_env_var('BINANCE_API_KEY')
BINANCE_SECRET_KEY = get_env_var('BINANCE_SECRET_KEY')
ALPHA_VANTAGE_API_KEY = get_env_var('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = get_env_var('NEWS_API_KEY')

# Trading Symbols - KEEPING YOUR EXACT SYMBOLS
CRYPTO_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT',
    'LINKUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT'
]

STOCK_SYMBOLS = [
    'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL',
    'NVDA', 'META', 'JPM', 'NFLX', 'AMD'
]

# Combine all symbols
ALL_SYMBOLS = CRYPTO_SYMBOLS + STOCK_SYMBOLS

# For backward compatibility - use ALL_SYMBOLS
SYMBOLS = ALL_SYMBOLS

# Asset Display Names - KEEPING YOUR EXACT NAMES
ASSET_DISPLAY_NAMES = {
    # Crypto
    'BTCUSDT': 'Bitcoin',
    'ETHUSDT': 'Ethereum',
    'ADAUSDT': 'Cardano',
    'DOTUSDT': 'Polkadot',
    'LINKUSDT': 'Chainlink',
    'SOLUSDT': 'Solana',
    'XRPUSDT': 'Ripple',
    'DOGEUSDT': 'Dogecoin',

    # Stocks
    'AAPL': 'Apple Inc.',
    'TSLA': 'Tesla Inc.',
    'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.',
    'GOOGL': 'Alphabet Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'NFLX': 'Netflix Inc.',
    'AMD': 'Advanced Micro Devices Inc.'
}

# Social media sources for sentiment (for backward compatibility)
SUBREDDITS = ['stocks', 'investing', 'wallstreetbets', 'CryptoCurrency', 'binance', 'CryptoMarkets', 'technology']

# Reddit Configuration (new structure)
REDDIT_SUBREDDITS = ['CryptoCurrency', 'stocks', 'investing', 'wallstreetbets', 'CryptoMarkets']

# News sources (for backward compatibility)
NEWS_SOURCES = ['bloomberg', 'reuters', 'financial-times', 'cnbc', 'the-wall-street-journal']

# News Configuration (new structure)
NEWS_SOURCES = ['bloomberg', 'reuters', 'financial-post', 'the-wall-street-journal']

# Keywords (for backward compatibility)
CRYPTO_KEYWORDS = ['bitcoin', 'ethereum', 'crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft']
STOCK_KEYWORDS = ['stocks', 'trading', 'investing', 'market', 'earnings', 'financial']

# App Configuration (for backward compatibility)
UPDATE_INTERVAL = 60  # seconds
SENTIMENT_THRESHOLD = 0.1

# Data Collection (new structure)
DATA_COLLECTION_INTERVAL = 3600  # 1 hour in seconds
SENTIMENT_UPDATE_INTERVAL = 1800  # 30 minutes in seconds

# Model Configuration
MODEL_SAVE_PATH = 'models/trained_models.pkl'
FEATURE_LOOKBACK_HOURS = 24
PREDICTION_HORIZON_HOURS = 1

# Database Configuration
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Deployment Configuration
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'development')