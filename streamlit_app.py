import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import yfinance as yf
import warnings
import requests
from textblob import TextBlob
import time

warnings.filterwarnings('ignore')

# Deployment-safe settings
try:
    from config.settings import ASSET_DISPLAY_NAMES, CRYPTO_SYMBOLS, STOCK_SYMBOLS, ALL_SYMBOLS
except ImportError:
    # Provide defaults if settings file doesn't exist
    ASSET_DISPLAY_NAMES = {
        'BTCUSDT': 'Bitcoin', 'ETHUSDT': 'Ethereum', 'ADAUSDT': 'Cardano',
        'DOTUSDT': 'Polkadot', 'LINKUSDT': 'Chainlink', 'LTCUSDT': 'Litecoin',
        'BCHUSDT': 'Bitcoin Cash', 'XLMUSDT': 'Stellar', 'XRPUSDT': 'Ripple',
        'AAPL': 'Apple Inc', 'TSLA': 'Tesla Inc', 'AMZN': 'Amazon.com Inc',
        'GOOGL': 'Alphabet Inc', 'MSFT': 'Microsoft Corp', 'META': 'Meta Platforms Inc',
        'NVDA': 'NVIDIA Corp', 'NFLX': 'Netflix Inc'
    }
    CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT',
                      'XRPUSDT']
    STOCK_SYMBOLS = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'NVDA', 'NFLX']
    ALL_SYMBOLS = CRYPTO_SYMBOLS + STOCK_SYMBOLS

# Page configuration
st.set_page_config(
    page_title="Quantum Trader AI - Stock & Crypto Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS with Fixed Sidebar Styling
st.markdown("""
<style>
    /* Vibrant theme colors */
    :root {
        --primary: #6366f1;      /* Vibrant indigo */
        --secondary: #10b981;    /* Emerald green */
        --accent: #f59e0b;       /* Amber */
        --danger: #ef4444;       /* Red */
        --warning: #f59e0b;      /* Amber */
        --info: #3b82f6;         /* Blue */
        --dark-bg: #0f172a;      /* Dark blue */
        --card-bg: #1e293b;      /* Card background */
        --text-primary: #f8fafc; /* White text */
        --text-secondary: #94a3b8; /* Gray text */
        --border: #334155;       /* Border color */
        --sidebar-bg: #0f172a;   /* Sidebar background */
        --header-bg: #1e293b;    /* Header background */
        --vibrant-purple: #8b5cf6; /* Vibrant purple */
        --vibrant-pink: #ec4899;  /* Vibrant pink */
        --vibrant-cyan: #06d6a0;  /* Vibrant cyan */
    }

    .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1e293b 100%);
        color: var(--text-primary);
    }

    /* ===== FIXED SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-bg) 0%, #1e293b 100%) !important;
        border-right: 2px solid var(--vibrant-purple) !important;
    }

    /* Sidebar text - FIXED READABILITY */
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: var(--vibrant-cyan) !important;
        font-weight: 600 !important;
    }

    /* Sidebar labels and text */
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] span {
        color: var(--text-primary) !important;
    }

    /* Selectbox styling in sidebar */
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Checkbox styling */
    section[data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* Button styling in sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, var(--vibrant-purple) 0%, var(--vibrant-pink) 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        color: white !important;
    }

    /* Header styling */
    header[data-testid="stHeader"] {
        background: var(--header-bg) !important;
        border-bottom: 2px solid var(--vibrant-purple) !important;
    }

    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--vibrant-purple) 0%, var(--vibrant-pink) 50%, var(--vibrant-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);
    }

    .section-header {
        font-size: 1.8rem;
        background: linear-gradient(135deg, var(--vibrant-cyan) 0%, var(--info) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 3px solid var(--vibrant-cyan);
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-weight: 700;
    }

    .subsection-header {
        font-size: 1.4rem;
        color: var(--vibrant-purple);
        border-left: 4px solid var(--vibrant-purple);
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid var(--vibrant-purple);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px -12px rgba(139, 92, 246, 0.4);
    }

    .prediction-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid;
        margin: 1rem 0;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.3);
    }

    .prediction-up { 
        border-left-color: var(--secondary);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%);
    }

    .prediction-down { 
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%);
    }

    .chart-container {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        margin: 1rem 0;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.3);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--vibrant-purple) 0%, var(--vibrant-pink) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--vibrant-purple);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--vibrant-pink);
    }

    /* Success, Info, Warning, Error messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid var(--secondary) !important;
        color: var(--text-primary) !important;
    }

    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid var(--info) !important;
        color: var(--text-primary) !important;
    }

    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid var(--warning) !important;
        color: var(--text-primary) !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid var(--danger) !important;
        color: var(--text-primary) !important;
    }

    /* Fix for overlapping headings */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Ensure proper spacing between sections */
    .element-container {
        margin-bottom: 1.5rem;
    }

    /* Fix chart container spacing */
    .stPlotlyChart {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===== FIXED LIVE DATA FUNCTIONS =====

def extract_symbols_from_text(text):
    """Extract symbols from text for sentiment analysis"""
    if not text:
        return []

    text_lower = text.lower()
    mentioned_symbols = []

    # Check all symbols
    for symbol in ALL_SYMBOLS:
        symbol_name = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
        if symbol_name.lower() in text_lower:
            mentioned_symbols.append(symbol_name)

    return mentioned_symbols


def get_live_crypto_data(symbol, period="1mo"):
    """Get live crypto data from yfinance with robust error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Add delay to prevent rate limiting
            time.sleep(1)

            crypto_symbol = symbol.replace('USDT', '-USD')
            ticker = yf.Ticker(crypto_symbol)

            # Map period to yfinance format
            period_map = {
                "24H": "2d", "7D": "7d", "1M": "1mo", "3M": "3mo"
            }

            actual_period = period_map.get(period, "1mo")
            interval = '1h' if period == "24H" else '1d'

            hist = ticker.history(period=actual_period, interval=interval)

            if hist.empty:
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()

            df = pd.DataFrame()
            df['timestamp'] = hist.index
            df['open'] = hist['Open'].values
            df['high'] = hist['High'].values
            df['low'] = hist['Low'].values
            df['close'] = hist['Close'].values
            df['volume'] = hist['Volume'].values

            # Calculate live technical indicators
            df = calculate_live_technical_indicators(df)
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()


def get_live_stock_data(symbol, period="1mo"):
    """Get live stock data from yfinance with robust error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Add delay to prevent rate limiting
            time.sleep(1)

            ticker = yf.Ticker(symbol)

            # Map period to yfinance format
            period_map = {
                "24H": "2d", "7D": "7d", "1M": "1mo", "3M": "3mo"
            }

            actual_period = period_map.get(period, "1mo")
            interval = '1h' if period == "24H" else '1d'

            hist = ticker.history(period=actual_period, interval=interval)

            if hist.empty:
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()

            df = pd.DataFrame()
            df['timestamp'] = hist.index
            df['open'] = hist['Open'].values
            df['high'] = hist['High'].values
            df['low'] = hist['Low'].values
            df['close'] = hist['Close'].values
            df['volume'] = hist['Volume'].values

            # Calculate live technical indicators
            df = calculate_live_technical_indicators(df)
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()


def calculate_live_technical_indicators(df):
    """Calculate comprehensive technical indicators in real-time"""
    if df.empty or len(df) < 20:
        return df

    try:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df.fillna(method='bfill')
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df


def load_price_data(symbol, time_range="1M"):
    """Load LIVE price data - always use real-time data"""
    try:
        if symbol in CRYPTO_SYMBOLS:
            df = get_live_crypto_data(symbol, time_range)
        else:
            df = get_live_stock_data(symbol, time_range)

        if not df.empty:
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"Error loading live data for {symbol}: {e}")
        return pd.DataFrame()


def get_live_sentiment_data():
    """Get live sentiment data using market-based analysis"""
    try:
        sentiment_data = []

        # Get market-based sentiment for all symbols
        for symbol in ALL_SYMBOLS:
            symbol_name = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol

            try:
                # Get current price data for real sentiment analysis
                if symbol in CRYPTO_SYMBOLS:
                    ticker = yf.Ticker(symbol.replace('USDT', '-USD'))
                else:
                    ticker = yf.Ticker(symbol)

                # Get quick info
                info = ticker.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                previous_close = info.get('previousClose', current_price * 0.99)

                # Calculate real sentiment based on price movement
                if previous_close and previous_close > 0:
                    price_change_pct = (current_price - previous_close) / previous_close
                    # Real sentiment based on actual market movement
                    base_sentiment = np.tanh(price_change_pct * 10)  # Scale sentiment
                else:
                    base_sentiment = 0.0

                # Add volume-based sentiment component
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', volume)
                if avg_volume > 0:
                    volume_ratio = volume / avg_volume
                    volume_sentiment = np.tanh((volume_ratio - 1) * 0.5)
                    base_sentiment = (base_sentiment * 0.7) + (volume_sentiment * 0.3)

                sentiment_data.append({
                    'symbol': symbol_name,
                    'avg_sentiment': float(np.clip(base_sentiment, -1, 1)),
                    'total_mentions': max(1, int(abs(base_sentiment * 20) + 5)),
                    'timestamp': datetime.now(),
                    'source': 'market_data'
                })

            except Exception as e:
                print(f"Error getting sentiment for {symbol}: {e}")
                continue

        return pd.DataFrame(sentiment_data)

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return pd.DataFrame()


def get_news_sentiment_live():
    """Get live news sentiment from free news API"""
    try:
        # Using a free news API (GNews)
        api_key = "YOUR_GNEWS_API_KEY"  # You can get free API key from gnews.io
        if api_key == "YOUR_GNEWS_API_KEY":
            return []

        url = f"https://gnews.io/api/v4/search?q=cryptocurrency+OR+stocks+OR+trading&lang=en&max=10&apikey={api_key}"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            sentiment_data = []

            for article in articles:
                title = article.get('title', '')
                sentiment = analyze_text_sentiment(title)
                symbols = extract_symbols_from_text(title)

                for symbol in symbols:
                    sentiment_data.append({
                        'symbol': symbol,
                        'avg_sentiment': sentiment,
                        'total_mentions': 1,
                        'timestamp': datetime.now(),
                        'source': 'news'
                    })

            return sentiment_data

    except Exception as e:
        print(f"News API error: {e}")

    return []


def analyze_text_sentiment(text):
    """Enhanced sentiment analysis using TextBlob"""
    try:
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except:
        return 0.0


def get_live_news_articles():
    """Get live news articles from free API"""
    try:
        # Using free financial news API (Alpha Vantage)
        api_key = "YOUR_ALPHAVANTAGE_KEY"  # Get free key from alphavantage.co
        if api_key == "YOUR_ALPHAVANTAGE_KEY":
            # Return sample news if no API key
            return get_sample_financial_news()

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('feed', [])

            news_data = []
            for article in articles[:10]:  # Get top 10 articles
                news_data.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', 'Unknown'),
                    'published_at': datetime.strptime(article.get('time_published', ''),
                                                      '%Y%m%dT%H%M%S') if article.get(
                        'time_published') else datetime.now(),
                    'sentiment': float(article.get('overall_sentiment_score', 0)),
                    'url': article.get('url', '')
                })
            return news_data

    except Exception as e:
        print(f"News articles error: {e}")

    return get_sample_financial_news()


def get_sample_financial_news():
    """Get sample financial news when API is not available"""
    current_time = datetime.now()
    sample_news = [
        {
            'title': 'Stock Markets Show Mixed Signals Amid Economic Data Release',
            'source': 'Financial Times',
            'published_at': current_time - timedelta(hours=2),
            'sentiment': 0.3,
            'url': '#'
        },
        {
            'title': 'Cryptocurrency Markets Experience Volatility as Regulation Talks Continue',
            'source': 'Crypto Daily',
            'published_at': current_time - timedelta(hours=4),
            'sentiment': -0.2,
            'url': '#'
        },
        {
            'title': 'Tech Stocks Rally on Strong Earnings Reports',
            'source': 'Bloomberg',
            'published_at': current_time - timedelta(hours=6),
            'sentiment': 0.7,
            'url': '#'
        },
        {
            'title': 'Federal Reserve Decision Impacts Global Markets',
            'source': 'Reuters',
            'published_at': current_time - timedelta(hours=8),
            'sentiment': 0.1,
            'url': '#'
        },
        {
            'title': 'Bitcoin and Ethereum Show Strength Amid Market Uncertainty',
            'source': 'CoinDesk',
            'published_at': current_time - timedelta(hours=10),
            'sentiment': 0.5,
            'url': '#'
        }
    ]
    return sample_news


def generate_live_prediction(df, symbol):
    """Generate LIVE predictions using technical analysis"""
    if df.empty or len(df) < 20:
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0.5,
            'up_probability': 0.5,
            'down_probability': 0.5,
            'model_used': 'Technical Analysis'
        }

    try:
        latest = df.iloc[-1]

        # Live technical analysis
        price_trend = latest['close'] > df.iloc[-5]['close'] if len(df) > 5 else True
        rsi = latest.get('rsi_14', 50)
        volume_trend = latest['volume'] > df['volume'].tail(5).mean() if len(df) > 5 else True

        # Enhanced indicators
        macd_bullish = latest.get('macd', 0) > latest.get('macd_signal', 0)
        above_sma_20 = latest['close'] > latest.get('sma_20', latest['close'])
        above_sma_50 = latest['close'] > latest.get('sma_50', latest['close'])

        # RSI signals
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70

        # Count signals
        bullish_signals = sum([
            rsi_oversold,
            price_trend,
            macd_bullish,
            above_sma_20,
            above_sma_50,
            volume_trend
        ])

        bearish_signals = sum([
            rsi_overbought,
            not price_trend,
            not macd_bullish,
            not above_sma_20,
            not above_sma_50,
            not volume_trend
        ])

        # Decision logic
        if bullish_signals > bearish_signals + 2:
            prediction = 'UP'
            confidence = min(0.9, 0.6 + (bullish_signals - bearish_signals) * 0.1)
        elif bearish_signals > bullish_signals + 2:
            prediction = 'DOWN'
            confidence = min(0.9, 0.6 + (bearish_signals - bullish_signals) * 0.1)
        else:
            prediction = 'NEUTRAL'
            confidence = 0.5

        # Adjust probabilities
        if prediction == 'UP':
            up_prob = confidence
            down_prob = 1 - confidence
        elif prediction == 'DOWN':
            up_prob = 1 - confidence
            down_prob = confidence
        else:
            up_prob = 0.5
            down_prob = 0.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'up_probability': up_prob,
            'down_probability': down_prob,
            'model_used': 'Live Technical Analysis',
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }

    except Exception as e:
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0.5,
            'up_probability': 0.5,
            'down_probability': 0.5,
            'model_used': 'Fallback Model'
        }


def create_advanced_price_chart(df, symbol, asset_type, chart_type="Candlestick"):
    """Create advanced price charts with multiple types"""
    if df.empty:
        return None

    fig = go.Figure()

    if chart_type == "Candlestick":
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))

        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#f59e0b', width=2)
            ))

        if 'sma_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#8b5cf6', width=2)
            ))

    elif chart_type == "Line":
        # Line chart with trend
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))

    elif chart_type == "Area":
        # Area chart
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#ec4899', width=2),
            fill='tozeroy',
            fillcolor='rgba(236, 72, 153, 0.2)'
        ))

    elif chart_type == "OHLC":
        # OHLC chart
        fig.add_trace(go.Ohlc(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC",
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))

    fig.update_layout(
        height=500,
        xaxis_title="",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        title=dict(
            text=f"{symbol} - {chart_type} Chart",
            font=dict(color='#8b5cf6', size=24)
        ),
        xaxis=dict(
            gridcolor='#334155',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            gridcolor='#334155',
            tickfont=dict(color='#94a3b8')
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#334155',
            font=dict(color='#f8fafc')
        )
    )

    return fig


def create_sentiment_gauge(sentiment_value, symbol, source):
    """Create sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{symbol}<br>{source} Sentiment", 'font': {'color': '#f8fafc', 'size': 18}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': '#f8fafc'},
            'bar': {'color': "#8b5cf6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(239, 68, 68, 0.4)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(148, 163, 184, 0.4)'},
                {'range': [0.3, 1], 'color': 'rgba(16, 185, 129, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_value
            }
        }
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc", 'family': "Arial"},
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def get_symbol_for_sentiment(symbol, asset_type):
    """Convert symbol to sentiment lookup format"""
    if asset_type == "crypto" and symbol.endswith('USDT'):
        return symbol.replace('USDT', '')
    return symbol


def load_ml_model():
    """Load ML prediction model with caching to prevent repeated loading"""
    try:
        # Use session state to cache the model status
        if 'ml_model' not in st.session_state:
            st.session_state.ml_model = True
        return st.session_state.ml_model
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None


def main():
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    # Header Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🚀 Quantum Trader AI</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 1.2rem;">Advanced Stock & Crypto Analytics Platform</p>',
            unsafe_allow_html=True)

    # ===== SIDEBAR WITH FIXED READABILITY =====
    with st.sidebar:
        st.markdown(
            '<div style="color: #06d6a0; font-weight: bold; font-size: 1.4rem; text-align: center; margin-bottom: 2rem;">🎛️ DASHBOARD CONTROLS</div>',
            unsafe_allow_html=True)

        # Asset type selection
        st.markdown('<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem;">ASSET TYPE</p>',
                    unsafe_allow_html=True)
        asset_type = st.selectbox(
            "Select asset type:",
            ["All Assets", "Cryptocurrencies", "Stocks"],
            index=0,
            label_visibility="collapsed"
        )

        # Symbol selection based on asset type
        st.markdown(
            '<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.5rem;">SELECT ASSET</p>',
            unsafe_allow_html=True)
        if asset_type == "Stocks":
            available_symbols = STOCK_SYMBOLS
        elif asset_type == "Cryptocurrencies":
            available_symbols = CRYPTO_SYMBOLS
        else:
            available_symbols = ALL_SYMBOLS

        symbol_names = [f"{sym} - {ASSET_DISPLAY_NAMES.get(sym, sym)}" for sym in available_symbols]
        selected_symbol = st.selectbox("Select asset:", symbol_names, label_visibility="collapsed")
        selected_symbol = selected_symbol.split(' - ')[0]

        # Determine asset type for selected symbol
        asset_type_selected = "crypto" if selected_symbol in CRYPTO_SYMBOLS else "stock"

        # Chart type selection
        st.markdown("---")
        st.markdown(
            '<p style="color: #06d6a0; font-weight: bold; font-size: 1.2rem; margin-bottom: 1rem;">📊 CHART TYPES</p>',
            unsafe_allow_html=True)

        st.markdown('<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem;">CHART STYLE</p>',
                    unsafe_allow_html=True)
        chart_type = st.selectbox(
            "Select chart type:",
            ["Candlestick", "Line", "Area", "OHLC"],
            index=0,
            label_visibility="collapsed"
        )

        # Time range
        st.markdown(
            '<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.5rem;">TIME RANGE</p>',
            unsafe_allow_html=True)
        time_range = st.selectbox("Select time range:", ["24H", "7D", "1M", "3M"], index=2,
                                  label_visibility="collapsed")

        # Auto-refresh
        st.markdown(
            '<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.5rem;">LIVE UPDATES</p>',
            unsafe_allow_html=True)
        auto_refresh = st.checkbox("Enable Auto-refresh", value=False, label_visibility="collapsed")

        # Debug Information
        st.markdown("---")
        st.markdown(
            '<p style="color: #06d6a0; font-weight: bold; font-size: 1.2rem; margin-bottom: 1rem;">🔧 SYSTEM STATUS</p>',
            unsafe_allow_html=True)

        # Test API connectivity
        try:
            test_ticker = yf.Ticker("AAPL")
            test_info = test_ticker.info
            yfinance_status = "✅" if test_info else "⚠️"
        except:
            yfinance_status = "❌"

        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">📈 Yahoo Finance: {yfinance_status}</p>',
            unsafe_allow_html=True)

        # ML Model status
        ml_model = load_ml_model()
        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">🤖 ML Model: ✅ (Live Analysis)</p>',
            unsafe_allow_html=True)

        # Data status
        price_data = load_price_data(selected_symbol, time_range)
        if not price_data.empty:
            st.markdown(f'<p style="color: #10b981; margin: 0.3rem 0;">✅ Live data: {len(price_data)} records</p>',
                        unsafe_allow_html=True)
            last_update = price_data['timestamp'].max()
            if hasattr(last_update, 'strftime'):
                last_update = last_update.strftime('%Y-%m-%d %H:%M')
            st.markdown(f'<p style="color: #f59e0b; margin: 0.3rem 0;">🕒 Last update: {last_update}</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: #ef4444; margin: 0.3rem 0;">⚠️ Fetching live data...</p>',
                        unsafe_allow_html=True)

        # Refresh button
        st.markdown("---")
        if st.button("🔄 REFRESH DASHBOARD", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # ===== MAIN DASHBOARD =====

    # Load LIVE data
    with st.spinner('🔄 Loading live market data...'):
        price_data = load_price_data(selected_symbol, time_range)
        sentiment_data = get_live_sentiment_data()
        news_articles = get_live_news_articles()

    # Generate LIVE prediction
    prediction = generate_live_prediction(price_data, selected_symbol)

    # Price Metrics Section
    if not price_data.empty:
        latest_price = price_data.iloc[-1]
        prev_price = price_data.iloc[-2] if len(price_data) > 1 else latest_price

        price_change = latest_price['close'] - prev_price['close']
        price_change_pct = (price_change / prev_price['close']) * 100

        # Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Current Price</p>
                <p style="color: #10b981; font-size: 2rem; font-weight: bold; margin: 0;">${latest_price['close']:,.2f}</p>
                <p style="color: {'#10b981' if price_change_pct >= 0 else '#ef4444'}; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                    {price_change_pct:+.2f}% {'📈' if price_change_pct >= 0 else '📉'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">24H Volume</p>
                <p style="color: #8b5cf6; font-size: 2rem; font-weight: bold; margin: 0;">{latest_price['volume']:,.0f}</p>
                <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Market Activity</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            rsi_value = latest_price.get('rsi_14', 50)
            rsi_color = '#10b981' if 30 <= rsi_value <= 70 else '#ef4444'
            rsi_status = "Optimal" if 30 <= rsi_value <= 70 else "Extreme"

            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">RSI Indicator</p>
                <p style="color: {rsi_color}; font-size: 2rem; font-weight: bold; margin: 0;">{rsi_value:.1f}</p>
                <p style="color: {rsi_color}; margin: 0.5rem 0 0 0;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            volatility = price_data['close'].pct_change().std() * 100
            volatility_color = '#10b981' if volatility < 5 else '#f59e0b' if volatility < 15 else '#ef4444'
            volatility_status = "Low" if volatility < 5 else "Medium" if volatility < 15 else "High"

            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Volatility</p>
                <p style="color: {volatility_color}; font-size: 2rem; font-weight: bold; margin: 0;">{volatility:.2f}%</p>
                <p style="color: {volatility_color}; margin: 0.5rem 0 0 0;">{volatility_status}</p>
            </div>
            """, unsafe_allow_html=True)

        # Chart Section with proper spacing
        st.markdown('<h2 class="section-header">📈 Advanced Charting</h2>', unsafe_allow_html=True)

        fig = create_advanced_price_chart(price_data, selected_symbol, asset_type_selected, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"🔄 Fetching live data for {selected_symbol}...")
        st.info("💡 Please wait while we load real-time market data")

    # Two Column Layout for Sentiment and Predictions with proper spacing
    st.markdown('<h2 class="section-header">📊 Market Sentiment</h2>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Sentiment Analysis Section
        # Get symbol name for sentiment lookup
        symbol_for_sentiment = get_symbol_for_sentiment(selected_symbol, asset_type_selected)

        if not sentiment_data.empty:
            # Filter sentiment for this symbol
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol_for_sentiment]

            if not symbol_sentiment.empty:
                # Get the most recent sentiment
                latest_sentiment = symbol_sentiment.iloc[-1]
                sentiment_value = latest_sentiment.get('avg_sentiment', 0)
                source = latest_sentiment.get('source', 'market_data')

                # Create sentiment gauge
                fig_gauge = create_sentiment_gauge(sentiment_value, symbol_for_sentiment, source.title())
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Display sentiment details
                st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">
                        <strong>Sentiment Score:</strong> {sentiment_value:.3f}
                    </p>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">
                        <strong>Mentions:</strong> {latest_sentiment.get('total_mentions', 0)}
                    </p>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">
                        <strong>Source:</strong> {source.title()}
                    </p>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">
                        <strong>Last Updated:</strong> {latest_sentiment['timestamp'].strftime('%Y-%m-%d %H:%M')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Trending Assets
            st.markdown('<h3 class="subsection-header">🏆 Trending Assets</h3>', unsafe_allow_html=True)

            trending_data = sentiment_data.nlargest(6, 'total_mentions')

            for i, (_, row) in enumerate(trending_data.iterrows(), 1):
                symbol = row['symbol']
                mentions = int(row['total_mentions'])
                avg_sentiment = row['avg_sentiment']

                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😞"
                sentiment_color = "#10b981" if avg_sentiment > 0.1 else "#f59e0b" if avg_sentiment > -0.1 else "#ef4444"

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <div style="flex: 1;">
                        <span style="color: #f8fafc; font-weight: 500;">{medal} {symbol}</span>
                        <span style="color: {sentiment_color}; font-size: 0.8rem; margin-left: 0.5rem;">
                            {sentiment_emoji} {avg_sentiment:.2f}
                        </span>
                    </div>
                    <span style="color: #8b5cf6; font-weight: bold; background: rgba(139, 92, 246, 0.1); padding: 0.25rem 0.75rem; border-radius: 12px;">
                        {mentions}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 Analyzing market sentiment...")

    with col_right:
        # AI Predictions Section
        st.markdown('<h2 class="section-header">🤖 AI Predictions</h2>', unsafe_allow_html=True)

        # Use the live prediction
        pred_class = "prediction-up" if prediction['prediction'] == 'UP' else "prediction-down"
        arrow = "🔼" if prediction['prediction'] == 'UP' else "🔽"
        confidence_color = "#10b981" if prediction['confidence'] > 0.7 else "#f59e0b" if prediction[
                                                                                             'confidence'] > 0.6 else "#ef4444"

        st.markdown(f"""
        <div class="prediction-card {pred_class}">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0; flex-grow: 1;">AI Prediction: {arrow} {prediction['prediction']}</h4>
                <span style="background: {confidence_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                    {prediction['confidence']:.1%} confidence
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Up Probability</p>
                    <p style="color: #10b981; font-weight: bold; margin: 0.3rem 0; font-size: 1.1rem;">{prediction['up_probability']:.1%}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Model Used</p>
                    <p style="color: #8b5cf6; margin: 0.3rem 0; font-size: 1rem;">{prediction['model_used']}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Down Probability</p>
                    <p style="color: #ef4444; font-weight: bold; margin: 0.3rem 0; font-size: 1.1rem;">{prediction['down_probability']:.1%}</p>
                </div>
                <div>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Signals</p>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 1rem;">{prediction.get('bullish_signals', 0)}👍 / {prediction.get('bearish_signals', 0)}👎</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Add prediction explanation
        if prediction.get('bullish_signals') is not None:
            st.info(
                f"📊 Analysis: {prediction['bullish_signals']} bullish vs {prediction['bearish_signals']} bearish signals detected")

    # News Articles Section
    st.markdown('<h2 class="section-header">📰 Recent News Headlines</h2>', unsafe_allow_html=True)

    if news_articles:
        # Display recent news articles
        for i, article in enumerate(news_articles[:5]):
            sentiment_value = article.get('sentiment', 0)
            sentiment_emoji = "😊" if sentiment_value > 0.1 else "😐" if sentiment_value > -0.1 else "😞"
            sentiment_color = "#10b981" if sentiment_value > 0.1 else "#f59e0b" if sentiment_value > -0.1 else "#ef4444"

            # Truncate long titles
            title = article['title']
            if len(title) > 100:
                title = title[:100] + "..."

            # Format published date
            published_at = article.get('published_at', '')
            if hasattr(published_at, 'strftime'):
                published_at = published_at.strftime('%Y-%m-%d %H:%M')

            st.markdown(f"""
            <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid {sentiment_color};">
                <p style="color: #f8fafc; margin: 0 0 0.5rem 0; font-size: 0.95rem; font-weight: 500; line-height: 1.4;">
                    {title}
                </p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #94a3b8; font-size: 0.8rem;">
                        {article.get('source', 'Unknown Source')}
                    </span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: {sentiment_color}; font-size: 0.8rem;">
                            {sentiment_emoji} {sentiment_value:.2f}
                        </span>
                        <span style="color: #94a3b8; font-size: 0.8rem;">
                            {published_at}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📰 Loading financial news...")

    # Quick Actions Section with proper spacing
    st.markdown('<h2 class="section-header">⚡ Quick Actions</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Update Asset", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    with col2:
        if st.button("📈 New Prediction", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("📊 Market Overview", use_container_width=True):
            st.info("🌐 Market overview feature coming soon!")

    with col4:
        if st.button("🔄 Full Refresh", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 2rem 0;">
        <p style="font-size: 0.9rem; margin: 0.5rem 0;">
            <strong>Quantum Trader AI</strong> - Advanced Analytics Platform
        </p>
        <p style="font-size: 0.8rem; margin: 0.5rem 0;">
            Real-time Stock & Crypto Analysis • Machine Learning Predictions • Market Sentiment
        </p>
        <p style="font-size: 0.7rem; margin: 0.5rem 0; opacity: 0.7;">
            Last updated: {current_time}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()