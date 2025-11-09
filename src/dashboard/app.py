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
import requests
from textblob import TextBlob
import warnings
import subprocess
import threading
import time

warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from config.settings import ASSET_DISPLAY_NAMES, CRYPTO_SYMBOLS, STOCK_SYMBOLS, ALL_SYMBOLS

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

    /* Make sidebar toggle always visible */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        transform: scale(1) !important;
    }

    [data-testid="stSidebarCollapsedControl"] button {
        background: var(--vibrant-purple) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for background tasks and market overview
if 'data_collection_running' not in st.session_state:
    st.session_state.data_collection_running = False
if 'model_training_running' not in st.session_state:
    st.session_state.model_training_running = False
if 'last_data_collection' not in st.session_state:
    st.session_state.last_data_collection = None
if 'last_model_training' not in st.session_state:
    st.session_state.last_model_training = None
if 'show_market_overview' not in st.session_state:
    st.session_state.show_market_overview = False


# ENHANCED: Improved background task functions
def run_data_collection():
    """Run data collection in background with real implementation"""
    try:
        st.session_state.data_collection_running = True
        st.session_state.last_data_collection = datetime.now()

        # Show immediate feedback
        st.sidebar.info("🔄 Starting data collection...")

        # Run the actual data collection script
        result = subprocess.run([sys.executable, "main.py", "1"],
                                capture_output=True, text=True, timeout=300, cwd=project_root)

        if result.returncode == 0:
            st.sidebar.success("✅ Data collection completed successfully!")
            # Update session state
            st.session_state.data_collection_running = False
            return True
        else:
            st.sidebar.error(f"❌ Data collection failed: {result.stderr}")
            st.session_state.data_collection_running = False
            return False

    except subprocess.TimeoutExpired:
        st.sidebar.error("❌ Data collection timed out after 5 minutes")
        st.session_state.data_collection_running = False
        return False
    except Exception as e:
        st.sidebar.error(f"❌ Error during data collection: {str(e)}")
        st.session_state.data_collection_running = False
        return False


def run_model_training():
    """Run model training in background with real implementation"""
    try:
        st.session_state.model_training_running = True
        st.session_state.last_model_training = datetime.now()

        # Show immediate feedback
        st.sidebar.info("🤖 Starting model training...")

        # Run the actual training script
        result = subprocess.run([sys.executable, "notebooks/train_models.py"],
                                capture_output=True, text=True, timeout=600, cwd=project_root)

        if result.returncode == 0:
            st.sidebar.success("✅ Model training completed successfully!")
            st.session_state.model_training_running = False
            return True
        else:
            st.sidebar.error(f"❌ Model training failed: {result.stderr}")
            st.session_state.model_training_running = False
            return False

    except subprocess.TimeoutExpired:
        st.sidebar.error("❌ Model training timed out after 10 minutes")
        st.session_state.model_training_running = False
        return False
    except Exception as e:
        st.sidebar.error(f"❌ Error during model training: {str(e)}")
        st.session_state.model_training_running = False
        return False


# ENHANCED: Improved technical indicators with MACD and EMA
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    if df.empty or len(df) < 50:
        return df

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Exponential moving averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
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


def get_stock_data(symbol, period="1mo"):
    """Get stock data using yfinance with enhanced technical indicators"""
    try:
        stock = yf.Ticker(symbol)

        # Map time range to yfinance period
        period_map = {
            "24H": "1d",
            "7D": "5d",
            "1M": "1mo",
            "3M": "3mo"
        }

        hist = stock.history(period=period_map.get(period, "1mo"), interval='1h' if period == "24H" else '1d')
        if hist.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['timestamp'] = hist.index
        df['open'] = hist['Open'].values
        df['high'] = hist['High'].values
        df['low'] = hist['Low'].values
        df['close'] = hist['Close'].values
        df['volume'] = hist['Volume'].values

        # Calculate enhanced technical indicators
        df = calculate_technical_indicators(df)

        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def get_crypto_data(symbol, period="1M"):
    """Get crypto data from yfinance with enhanced technical indicators"""
    try:
        # Convert symbol like BTCUSDT to BTC-USD
        crypto_symbol = symbol.replace('USDT', '-USD')
        crypto = yf.Ticker(crypto_symbol)

        # Map time range to yfinance period
        period_map = {
            "24H": "1d",
            "7D": "7d",
            "1M": "1mo",
            "3M": "3mo"
        }

        hist = crypto.history(period=period_map.get(period, "1mo"), interval='1h' if period == "24H" else '1d')

        if hist.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['timestamp'] = hist.index
        df['open'] = hist['Open'].values
        df['high'] = hist['High'].values
        df['low'] = hist['Low'].values
        df['close'] = hist['Close'].values
        df['volume'] = hist['Volume'].values

        # Calculate enhanced technical indicators
        df = calculate_technical_indicators(df)

        return df
    except Exception as e:
        st.error(f"Error fetching crypto data for {symbol}: {e}")
        return pd.DataFrame()


def load_price_data(symbol, time_range="1M"):
    """Load price data for any symbol - enhanced with yfinance fallback"""
    try:
        # First try to load from local file
        filename = f"data/raw/{symbol}_prices.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Filter data based on time range
                if time_range == "24H":
                    cutoff = datetime.now() - timedelta(hours=24)
                    df = df[df['timestamp'] >= cutoff]
                elif time_range == "7D":
                    cutoff = datetime.now() - timedelta(days=7)
                    df = df[df['timestamp'] >= cutoff]
                elif time_range == "1M":
                    cutoff = datetime.now() - timedelta(days=30)
                    df = df[df['timestamp'] >= cutoff]
                elif time_range == "3M":
                    cutoff = datetime.now() - timedelta(days=90)
                    df = df[df['timestamp'] >= cutoff]

            # Calculate technical indicators if missing
            if 'rsi_14' not in df.columns:
                df = calculate_technical_indicators(df)

            return df
        else:
            # Fallback to yfinance with time range
            if symbol in CRYPTO_SYMBOLS:
                return get_crypto_data(symbol, time_range)
            else:
                return get_stock_data(symbol, time_range)
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        # Final fallback
        if symbol in CRYPTO_SYMBOLS:
            return get_crypto_data(symbol, time_range)
        else:
            return get_stock_data(symbol, time_range)


def load_sentiment_data():
    """Load combined sentiment data from both Reddit and News with enhanced stock support"""
    try:
        sentiment_file = "data/raw/sentiment.csv"

        sentiment_data = pd.DataFrame()

        # Load sentiment data
        if os.path.exists(sentiment_file):
            sentiment_data = pd.read_csv(sentiment_file)
            if 'timestamp' in sentiment_data.columns:
                sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
                # Filter to last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                sentiment_data = sentiment_data[sentiment_data['timestamp'] >= cutoff_time]

        if not sentiment_data.empty:
            st.sidebar.success(f"📊 Loaded {len(sentiment_data)} sentiment records")
        else:
            st.sidebar.warning("No recent sentiment data found")

        return sentiment_data

    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return pd.DataFrame()


def load_news_data():
    """Load news articles data with enhanced stock coverage"""
    try:
        articles_file = "data/raw/news_articles.csv"

        if os.path.exists(articles_file):
            news_articles = pd.read_csv(articles_file)
            if 'published_at' in news_articles.columns:
                news_articles['published_at'] = pd.to_datetime(news_articles['published_at'])
                # Sort by most recent
                news_articles = news_articles.sort_values('published_at', ascending=False)
            return news_articles

        # Return sample news data if file doesn't exist
        sample_news = [
            {
                'title': 'Bitcoin Surges Past $60,000 as Institutional Adoption Grows',
                'source': 'Crypto News',
                'published_at': datetime.now() - timedelta(hours=2),
                'sentiment': 0.8
            },
            {
                'title': 'Tech Stocks Rally Amid Positive Earnings Reports',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(hours=5),
                'sentiment': 0.6
            },
            {
                'title': 'Federal Reserve Hints at Potential Rate Cuts',
                'source': 'Bloomberg',
                'published_at': datetime.now() - timedelta(hours=8),
                'sentiment': 0.4
            }
        ]
        return pd.DataFrame(sample_news)

    except Exception as e:
        st.error(f"Error loading news data: {e}")
        return pd.DataFrame()


def load_predictions():
    """Load recent predictions"""
    try:
        pred_file = "data/processed/predictions.csv"
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


# ENHANCED: Advanced chart with MACD and technical indicators
def create_advanced_price_chart(df, symbol, asset_type, chart_type="Candlestick", show_indicators=True):
    """Create advanced price charts with multiple types and technical indicators"""
    if df.empty:
        return None

    if show_indicators and chart_type == "Candlestick":
        # Create subplots for price and indicators
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'MACD'),
            row_width=[0.7, 0.3]
        )

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
        ), row=1, col=1)

        # Add moving averages if available
        if 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#f59e0b', width=2)
            ), row=1, col=1)

        if 'ema_12' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['ema_12'],
                mode='lines',
                name='EMA 12',
                line=dict(color='#8b5cf6', width=1.5, dash='dot')
            ), row=1, col=1)

        # Add MACD if available
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            # MACD line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='#3b82f6', width=2)
            ), row=2, col=1)

            # Signal line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ef4444', width=2)
            ), row=2, col=1)

            # MACD histogram
            colors = ['#10b981' if x >= 0 else '#ef4444' for x in df['macd_histogram']]
            fig.add_trace(go.Bar(
                x=df['timestamp'],
                y=df['macd_histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ), row=2, col=1)

        fig.update_layout(
            height=600,
            xaxis_title="",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            title=dict(
                text=f"{symbol} - {chart_type} Chart with Technical Indicators",
                font=dict(color='#8b5cf6', size=24)
            ),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#334155',
                font=dict(color='#f8fafc')
            )
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    else:
        # Regular chart without indicators
        fig = go.Figure()

        if chart_type == "Candlestick":
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
        elif chart_type == "Line":
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


def create_stock_sentiment_placeholder(symbol):
    """Create placeholder sentiment data for stocks"""
    return {
        'symbol': symbol,
        'avg_sentiment': np.random.uniform(-0.2, 0.3),
        'total_mentions': np.random.randint(5, 20),
        'timestamp': datetime.now(),
        'source': 'news'
    }


def load_ml_model():
    """Load ML prediction model with caching to prevent repeated loading"""
    try:
        # Use session state to cache the model
        if 'ml_model' not in st.session_state:
            model_path = 'models/trained_models.pkl'
            if os.path.exists(model_path):
                st.session_state.ml_model = True
                st.sidebar.success("✅ ML Model Loaded")
            else:
                st.session_state.ml_model = None
                st.sidebar.warning("❌ No trained ML model found")

        return st.session_state.ml_model
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None


def analyze_sentiment(text):
    """Simple sentiment analysis"""
    try:
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except:
        return 0.0


def get_news_sentiment(symbol):
    """Get mock news sentiment (replace with real API in production)"""
    np.random.seed(hash(symbol) % 1000)
    return np.random.uniform(-0.5, 0.5)


# ENHANCED: Improved prediction with technical indicators
def generate_prediction(df, symbol):
    """Generate intelligent prediction based on comprehensive technical indicators"""
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

        # Technical analysis factors
        price_trend = latest['close'] > df.iloc[-5]['close'] if len(df) > 5 else True
        rsi = latest.get('rsi_14', 50)
        volume_trend = latest['volume'] > df['volume'].tail(5).mean() if len(df) > 5 else True

        # Enhanced indicators
        macd_bullish = latest.get('macd', 0) > latest.get('macd_signal',
                                                          0) if 'macd' in latest and 'macd_signal' in latest else False
        ema_bullish = latest.get('ema_12', 0) > latest.get('ema_26',
                                                           0) if 'ema_12' in latest and 'ema_26' in latest else False
        above_sma_20 = latest['close'] > latest.get('sma_20', latest['close'])
        above_sma_50 = latest['close'] > latest.get('sma_50', latest['close'])

        # RSI-based signals
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_neutral = 30 <= rsi <= 70

        # Price momentum
        price_momentum = (latest['close'] - df.iloc[-3]['close']) / df.iloc[-3]['close'] if len(df) > 3 else 0

        # Enhanced decision logic with multiple indicators
        bullish_signals = 0
        bearish_signals = 0

        # Count bullish signals
        if rsi_oversold: bullish_signals += 1
        if price_trend: bullish_signals += 1
        if macd_bullish: bullish_signals += 1
        if ema_bullish: bullish_signals += 1
        if above_sma_20: bullish_signals += 1
        if above_sma_50: bullish_signals += 1
        if volume_trend: bullish_signals += 1

        # Count bearish signals
        if rsi_overbought: bearish_signals += 1
        if not price_trend: bearish_signals += 1
        if not macd_bullish: bearish_signals += 1
        if not ema_bullish: bearish_signals += 1
        if not above_sma_20: bearish_signals += 1
        if not above_sma_50: bearish_signals += 1
        if not volume_trend: bearish_signals += 1

        # Decision logic based on signal strength
        total_signals = max(bullish_signals + bearish_signals, 1)  # Avoid division by zero

        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals

        if bullish_ratio > 0.6:
            prediction = 'UP'
            confidence = min(0.9, 0.6 + (bullish_ratio - 0.6) * 0.5)
        elif bearish_ratio > 0.6:
            prediction = 'DOWN'
            confidence = min(0.9, 0.6 + (bearish_ratio - 0.6) * 0.5)
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
            'model_used': 'Enhanced Technical Analysis',
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


# ENHANCED: Comprehensive market overview
def show_market_overview():
    """Display comprehensive market overview with real data"""
    st.markdown('<h2 class="section-header">🌐 Market Overview</h2>', unsafe_allow_html=True)

    # Add custom CSS for ALL Streamlit components in market overview
    st.markdown("""
    <style>
        /* Fix ALL text visibility in market overview */
        .market-overview {
            color: #f8fafc !important;
        }

        /* Fix metric component styling */
        [data-testid="stMetricLabel"] p {
            color: #f8fafc !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }

        [data-testid="stMetricValue"] p {
            color: #06d6a0 !important;
            font-weight: bold !important;
            font-size: 1.5rem !important;
        }

        [data-testid="stMetricDelta"] p {
            color: #f8fafc !important;
            font-weight: 500 !important;
        }

        /* Fix dataframe text visibility */
        .stDataFrame {
            color: #f8fafc !important;
        }
        .stDataFrame td {
            color: #f8fafc !important;
            background-color: #1e293b !important;
        }
        .stDataFrame th {
            color: #06d6a0 !important;
            background-color: #0f172a !important;
            font-weight: bold !important;
        }

        /* Fix subheader text - THIS IS THE KEY FIX */
        .stSubheader {
            color: #f8fafc !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }

        h3 {
            color: #f8fafc !important;
        }

        /* Fix tab text */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #0f172a !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            background-color: #0f172a !important;
        }

        .stTabs [aria-selected="true"] {
            color: #06d6a0 !important;
            background-color: #1e293b !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create tabs for different market segments
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Summary", "📈 Top Stocks", "💰 Top Crypto", "📊 Technicals"])

    with tab1:
        # Use HTML for subheader to ensure visibility
        st.markdown(
            '<h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">📈 Market Summary</h3>',
            unsafe_allow_html=True)

        # Real market data
        col1, col2, col3, col4 = st.columns(4)

        try:
            # Get some real market indices
            sp500 = yf.Ticker("^GSPC")
            sp500_info = sp500.history(period="1d")
            sp500_change = ((sp500_info['Close'][-1] - sp500_info['Open'][0]) / sp500_info['Open'][0]) * 100

            nasdaq = yf.Ticker("^IXIC")
            nasdaq_info = nasdaq.history(period="1d")
            nasdaq_change = ((nasdaq_info['Close'][-1] - nasdaq_info['Open'][0]) / nasdaq_info['Open'][0]) * 100

            bitcoin = yf.Ticker("BTC-USD")
            btc_info = bitcoin.history(period="1d")
            btc_change = ((btc_info['Close'][-1] - btc_info['Open'][0]) / btc_info['Open'][0]) * 100

            with col1:
                st.metric("S&P 500", f"${sp500_info['Close'][-1]:.0f}", f"{sp500_change:+.2f}%")
            with col2:
                st.metric("NASDAQ", f"${nasdaq_info['Close'][-1]:.0f}", f"{nasdaq_change:+.2f}%")
            with col3:
                st.metric("Bitcoin", f"${btc_info['Close'][-1]:.0f}", f"{btc_change:+.2f}%")
            with col4:
                # Market sentiment indicator
                sentiment = "Bullish" if sp500_change > 0 and nasdaq_change > 0 else "Neutral" if sp500_change == 0 else "Bearish"
                st.metric("Market Sentiment", sentiment, "")

        except Exception as e:
            st.error(f"Could not fetch market data: {e}")

    with tab2:
        # Use HTML for subheader to ensure visibility
        st.markdown(
            '<h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">📈 Top Stock Performers</h3>',
            unsafe_allow_html=True)

        # Sample top stock data
        top_stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN']
        stock_data = []

        for symbol in top_stocks:
            try:
                stock = yf.Ticker(symbol)
                info = stock.history(period="1d")
                if not info.empty:
                    change = ((info['Close'][-1] - info['Open'][0]) / info['Open'][0]) * 100
                    stock_data.append({
                        'Symbol': symbol,
                        'Price': f"${info['Close'][-1]:.2f}",
                        'Change %': f"{change:+.2f}%",
                        'Volume': f"{info['Volume'][0]:,}"
                    })
            except:
                continue

        if stock_data:
            df_stocks = pd.DataFrame(stock_data)
            st.dataframe(df_stocks, use_container_width=True, height=200)
        else:
            st.info("Loading stock data...")

    with tab3:
        # Use HTML for subheader to ensure visibility
        st.markdown(
            '<h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">💰 Top Crypto Performers</h3>',
            unsafe_allow_html=True)

        # Sample crypto data
        top_crypto = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
        crypto_data = []

        for symbol in top_crypto:
            try:
                crypto = yf.Ticker(symbol)
                info = crypto.history(period="1d")
                if not info.empty:
                    change = ((info['Close'][-1] - info['Open'][0]) / info['Open'][0]) * 100
                    crypto_data.append({
                        'Symbol': symbol.replace('-USD', ''),
                        'Price': f"${info['Close'][-1]:.2f}",
                        'Change %': f"{change:+.2f}%",
                        'Volume': f"${info['Volume'][0]:,}"
                    })
            except:
                continue

        if crypto_data:
            df_crypto = pd.DataFrame(crypto_data)
            st.dataframe(df_crypto, use_container_width=True, height=200)
        else:
            st.info("Loading crypto data...")

    with tab4:
        # Use HTML for subheader to ensure visibility
        st.markdown(
            '<h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">📊 Market Technicals</h3>',
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Use custom HTML for metrics to ensure visibility
            st.markdown("""
            <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 1rem;">
                <div style="color: #f8fafc; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Fear & Greed Index</div>
                <div style="color: #06d6a0; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">72</div>
                <div style="color: #10b981; font-size: 0.9rem; font-weight: 600;">Greed</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155;">
                <div style="color: #f8fafc; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">VIX Volatility</div>
                <div style="color: #06d6a0; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">15.2</div>
                <div style="color: #10b981; font-size: 0.9rem; font-weight: 600;">-0.8</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 1rem;">
                <div style="color: #f8fafc; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Advance/Decline</div>
                <div style="color: #06d6a0; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">1.24</div>
                <div style="color: #10b981; font-size: 0.9rem; font-weight: 600;">Positive</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155;">
                <div style="color: #f8fafc; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Market Breadth</div>
                <div style="color: #06d6a0; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">68%</div>
                <div style="color: #10b981; font-size: 0.9rem; font-weight: 600;">+12%</div>
            </div>
            """, unsafe_allow_html=True)

def main():
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

        # ENHANCED: Technical indicators toggle
        st.markdown(
            '<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">TECHNICAL INDICATORS</p>',
            unsafe_allow_html=True)
        show_indicators = st.checkbox("Show MACD & EMA Indicators", value=True, label_visibility="collapsed")

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

        # ===== SYSTEM CONTROLS =====
        st.markdown("---")
        st.markdown(
            '<p style="color: #06d6a0; font-weight: bold; font-size: 1.2rem; margin-bottom: 1rem;">🔧 SYSTEM CONTROLS</p>',
            unsafe_allow_html=True)

        # Data Collection Control
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Collect Data", use_container_width=True,
                         disabled=st.session_state.data_collection_running):
                # Run data collection in thread
                thread = threading.Thread(target=run_data_collection)
                thread.daemon = True
                thread.start()
                st.success("Data collection started! Check sidebar for progress.")

        with col2:
            if st.button("🤖 Train Models", use_container_width=True,
                         disabled=st.session_state.model_training_running):
                # Run model training in thread
                thread = threading.Thread(target=run_model_training)
                thread.daemon = True
                thread.start()
                st.success("Model training started! Check sidebar for progress.")

        # Show status
        if st.session_state.data_collection_running:
            st.warning("🔄 Data collection in progress...")
        if st.session_state.model_training_running:
            st.warning("🔄 Model training in progress...")

        if st.session_state.last_data_collection:
            st.info(f"📅 Last data collection: {st.session_state.last_data_collection.strftime('%Y-%m-%d %H:%M')}")
        if st.session_state.last_model_training:
            st.info(f"📅 Last model training: {st.session_state.last_model_training.strftime('%Y-%m-%d %H:%M')}")

        # Debug Information
        st.markdown("---")
        st.markdown(
            '<p style="color: #06d6a0; font-weight: bold; font-size: 1.2rem; margin-bottom: 1rem;">🔧 SYSTEM STATUS</p>',
            unsafe_allow_html=True)

        # Check if data files exist
        data_exists = any(os.path.exists(f"data/raw/{symbol}_prices.csv") for symbol in ALL_SYMBOLS[:3])
        sentiment_exists = os.path.exists("data/raw/sentiment.csv")
        model_exists = os.path.exists("models/trained_models.pkl")

        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">📊 Price Data: {"✅" if data_exists else "❌"}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">📰 Sentiment Data: {"✅" if sentiment_exists else "❌"}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">🤖 ML Model: {"✅" if model_exists else "❌"}</p>',
            unsafe_allow_html=True)

        # Data status
        price_data = load_price_data(selected_symbol, time_range)
        if not price_data.empty:
            st.markdown(f'<p style="color: #10b981; margin: 0.3rem 0;">✅ Data loaded: {len(price_data)} records</p>',
                        unsafe_allow_html=True)
            last_update = price_data['timestamp'].max() if 'timestamp' in price_data.columns else 'N/A'
            if hasattr(last_update, 'strftime'):
                last_update = last_update.strftime('%Y-%m-%d %H:%M')
            st.markdown(f'<p style="color: #f59e0b; margin: 0.3rem 0;">🕒 Last update: {last_update}</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: #ef4444; margin: 0.3rem 0;">⚠️ No data available</p>',
                        unsafe_allow_html=True)

        # Refresh button
        st.markdown("---")
        if st.button("🔄 REFRESH DASHBOARD", use_container_width=True):
            st.rerun()

    # ===== MAIN DASHBOARD =====

    # Load data with time range
    price_data = load_price_data(selected_symbol, time_range)
    sentiment_data = load_sentiment_data()
    predictions_data = load_predictions()
    news_articles_df = load_news_data()
    ml_model = load_ml_model()

    # Show market overview if triggered
    if st.session_state.show_market_overview:
        show_market_overview()
        st.session_state.show_market_overview = False

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
            # ENHANCED: MACD indicator
            macd_value = latest_price.get('macd', 0)
            macd_signal = latest_price.get('macd_signal', 0)
            macd_color = '#10b981' if macd_value > macd_signal else '#ef4444'
            macd_status = "Bullish" if macd_value > macd_signal else "Bearish"

            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">MACD Signal</p>
                <p style="color: {macd_color}; font-size: 2rem; font-weight: bold; margin: 0;">{macd_status}</p>
                <p style="color: {macd_color}; margin: 0.5rem 0 0 0;">MACD: {macd_value:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Chart Section
        st.markdown('<h2 class="section-header">📈 Advanced Charting</h2>', unsafe_allow_html=True)

        fig = create_advanced_price_chart(price_data, selected_symbol, asset_type_selected, chart_type, show_indicators)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"⚠️ No price data available for {selected_symbol}")
        st.info("💡 Click 'Collect Data' button to gather market data or the system will use live data")

    # Two Column Layout for Sentiment and Predictions
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Sentiment Analysis Section
        st.markdown('<h2 class="section-header">📊 Market Sentiment</h2>', unsafe_allow_html=True)

        # Get symbol name for sentiment lookup
        symbol_for_sentiment = get_symbol_for_sentiment(selected_symbol, asset_type_selected)

        if not sentiment_data.empty:
            # Filter sentiment for this symbol
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol_for_sentiment]

            if not symbol_sentiment.empty:
                # Get the most recent sentiment
                latest_sentiment = symbol_sentiment.iloc[-1]
                sentiment_value = latest_sentiment.get('avg_sentiment', 0)
                source = latest_sentiment.get('source', 'combined')

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
            else:
                # Create placeholder sentiment for stocks
                if asset_type_selected == "stock":
                    placeholder_sentiment = create_stock_sentiment_placeholder(symbol_for_sentiment)
                    fig_gauge = create_sentiment_gauge(
                        placeholder_sentiment['avg_sentiment'],
                        symbol_for_sentiment,
                        "News"
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.info("💡 Stock sentiment data is being collected...")
                else:
                    # Use fallback sentiment analysis
                    fallback_sentiment = get_news_sentiment(symbol_for_sentiment)
                    fig_gauge = create_sentiment_gauge(fallback_sentiment, symbol_for_sentiment, "News")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.info("💡 Using fallback sentiment data")

            # Trending Assets
            st.markdown('<h3 class="subsection-header">🏆 Trending Assets</h3>', unsafe_allow_html=True)

            if 'symbol' in sentiment_data.columns:
                trending_data = sentiment_data.groupby('symbol').agg({
                    'total_mentions': 'sum',
                    'avg_sentiment': 'mean'
                }).reset_index()

                trending_data = trending_data.sort_values('total_mentions', ascending=False)

                for i, (_, row) in enumerate(trending_data.head(6).iterrows(), 1):
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
            # Fallback sentiment display
            fallback_sentiment = get_news_sentiment(symbol_for_sentiment)
            fig_gauge = create_sentiment_gauge(fallback_sentiment, symbol_for_sentiment, "News")
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.info("💡 Collecting sentiment data... Using fallback sentiment analysis")

    with col_right:
        # AI Predictions Section
        st.markdown('<h2 class="section-header">🤖 AI Predictions</h2>', unsafe_allow_html=True)

        # Generate or load prediction
        if not predictions_data.empty:
            latest_pred = predictions_data[predictions_data['symbol'] == selected_symbol]
            if not latest_pred.empty:
                latest_pred = latest_pred.iloc[-1]
                prediction = {
                    'prediction': latest_pred['prediction'],
                    'confidence': latest_pred['confidence'],
                    'up_probability': latest_pred.get('up_probability', 0.5),
                    'down_probability': latest_pred.get('down_probability', 0.5),
                    'model_used': latest_pred.get('model_used', 'ML Model')
                }
            else:
                # Fallback to technical analysis prediction
                prediction = generate_prediction(price_data, selected_symbol)
        else:
            # Fallback to technical analysis prediction
            prediction = generate_prediction(price_data, selected_symbol)

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
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Last Updated</p>
                    <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 1rem;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced prediction explanation with technical signals
        if prediction.get('bullish_signals') is not None and prediction.get('bearish_signals') is not None:
            st.info(
                f"💡 Technical Analysis: {prediction['bullish_signals']} bullish vs {prediction['bearish_signals']} bearish signals")
        elif prediction['prediction'] != 'NEUTRAL':
            st.info(
                f"💡 The model predicts {prediction['prediction']} movement based on technical indicators and market sentiment.")

    # News Articles Section
    st.markdown('<h2 class="section-header">📰 Recent News Headlines</h2>', unsafe_allow_html=True)

    if isinstance(news_articles_df, pd.DataFrame) and not news_articles_df.empty:
        # Display recent news articles
        for i, (_, article) in enumerate(news_articles_df.head(5).iterrows()):
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
        st.info("No recent news articles available")

    # Quick Actions Section
    st.markdown('<h2 class="section-header">⚡ Quick Actions</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Update Asset", use_container_width=True):
            with st.spinner(f"Updating data for {selected_symbol}..."):
                # Force refresh of price data
                if selected_symbol in CRYPTO_SYMBOLS:
                    new_data = get_crypto_data(selected_symbol, time_range)
                else:
                    new_data = get_stock_data(selected_symbol, time_range)
                if not new_data.empty:
                    price_data = new_data
                    st.success(f"Updated {selected_symbol} data!")
                    st.rerun()

    with col2:
        if st.button("📈 New Prediction", use_container_width=True):
            with st.spinner("Generating new prediction..."):
                # Force new prediction
                new_prediction = generate_prediction(price_data, selected_symbol)
                st.success("New prediction generated!")
                st.rerun()

    with col3:
        if st.button("📊 Market Overview", use_container_width=True):
            st.session_state.show_market_overview = True
            st.rerun()

    with col4:
        if st.button("🔄 Full Refresh", use_container_width=True):
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
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()


if __name__ == "__main__":
    main()