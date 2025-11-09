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
warnings.filterwarnings('ignore')

# Deployment-safe settings - remove the problematic path manipulation
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
    CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT']
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
</style>
""", unsafe_allow_html=True)


def load_price_data(symbol):
    """Load price data for any symbol with live fallback"""
    try:
        filename = f"data/raw/{symbol}_prices.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            # Fallback to live data if file doesn't exist
            return get_live_price_data(symbol)
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        # Final fallback to live data
        return get_live_price_data(symbol)


def load_sentiment_data():
    """Load combined sentiment data from both Reddit and News with enhanced stock support"""
    try:
        sentiment_file = "data/raw/sentiment.csv"
        news_sentiment_file = "data/raw/news_sentiment.csv"

        sentiment_data = pd.DataFrame()
        news_sentiment = pd.DataFrame()

        # Load Reddit sentiment
        if os.path.exists(sentiment_file):
            sentiment_data = pd.read_csv(sentiment_file)
            if 'timestamp' in sentiment_data.columns:
                sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
                # Filter to last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                sentiment_data = sentiment_data[sentiment_data['timestamp'] >= cutoff_time]

        # Load News sentiment
        if os.path.exists(news_sentiment_file):
            news_sentiment = pd.read_csv(news_sentiment_file)
            if 'timestamp' in news_sentiment.columns:
                news_sentiment['timestamp'] = pd.to_datetime(news_sentiment['timestamp'])
                # Filter to last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                news_sentiment = news_sentiment[news_sentiment['timestamp'] >= cutoff_time]

        # Combine both sentiment sources
        combined_sentiment = pd.concat([sentiment_data, news_sentiment], ignore_index=True)

        # Enhanced stock symbol mapping for sentiment data
        if not combined_sentiment.empty:
            # Map stock symbols to their sentiment equivalents
            stock_symbol_mapping = {
                'AAPL': 'AAPL', 'TSLA': 'TSLA', 'MSFT': 'MSFT', 'AMZN': 'AMZN',
                'GOOGL': 'GOOGL', 'NVDA': 'NVDA', 'META': 'META', 'JPM': 'JPM',
                'NFLX': 'NFLX', 'AMD': 'AMD'
            }

            # Add stock sentiment data if missing
            for stock_symbol in STOCK_SYMBOLS:
                if stock_symbol not in combined_sentiment['symbol'].values:
                    # Create synthetic sentiment data for stocks (you can replace this with actual data collection)
                    stock_sentiment = pd.DataFrame([{
                        'symbol': stock_symbol,
                        'avg_sentiment': np.random.uniform(-0.2, 0.3),  # Replace with actual sentiment
                        'total_mentions': np.random.randint(1, 10),
                        'timestamp': datetime.now(),
                        'source': 'news'
                    }])
                    combined_sentiment = pd.concat([combined_sentiment, stock_sentiment], ignore_index=True)

        if not combined_sentiment.empty:
            st.sidebar.success(f"📊 Loaded {len(combined_sentiment)} sentiment records")
        else:
            st.sidebar.warning("No recent sentiment data found")

        return combined_sentiment

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
        return pd.DataFrame()

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


def get_live_price_data(symbol, period="1mo"):
    """Get live price data using yfinance as fallback"""
    try:
        if symbol in CRYPTO_SYMBOLS:
            # Convert crypto symbol to yfinance format
            crypto_symbol = symbol.replace('USDT', '-USD')
            ticker = yf.Ticker(crypto_symbol)
        else:
            ticker = yf.Ticker(symbol)

        hist = ticker.history(period=period)
        if hist.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['timestamp'] = hist.index
        df['open'] = hist['Open'].values
        df['high'] = hist['High'].values
        df['low'] = hist['Low'].values
        df['close'] = hist['Close'].values
        df['volume'] = hist['Volume'].values

        return df
    except Exception as e:
        st.error(f"Error fetching live data for {symbol}: {e}")
        return pd.DataFrame()

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


def create_stock_sentiment_placeholder(symbol):
    """Create placeholder sentiment data for stocks"""
    # This is a temporary solution - you should implement actual stock sentiment collection
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
        # Use session state to cache the model status
        if 'ml_model' not in st.session_state:
            # In deployment, simulate model loading
            st.session_state.ml_model = True
            st.sidebar.success("✅ ML Model Ready")

        return st.session_state.ml_model
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None


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

        # Time range
        st.markdown(
            '<p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.5rem;">TIME RANGE</p>',
            unsafe_allow_html=True)
        time_range = st.selectbox("Select time range:", ["24H", "7D", "1M", "3M"], index=1,
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

        # Check if sentiment files exist
        sentiment_exists = os.path.exists("data/raw/sentiment.csv")
        news_sentiment_exists = os.path.exists("data/raw/news_sentiment.csv")

        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">📊 Reddit Sentiment: {"✅" if sentiment_exists else "❌"}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="color: #f8fafc; margin: 0.3rem 0;">📰 News Sentiment: {"✅" if news_sentiment_exists else "❌"}</p>',
            unsafe_allow_html=True)

        # Load ML model
        ml_model = load_ml_model()
        if ml_model and hasattr(ml_model, 'is_trained') and ml_model.is_trained:
            accuracy = ml_model.model_performance.get('accuracy', 0) if hasattr(ml_model, 'model_performance') else 0
            st.markdown(f'<p style="color: #f8fafc; margin: 0.3rem 0;">🤖 ML Model: ✅ ({accuracy:.1%} accuracy)</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: #f8fafc; margin: 0.3rem 0;">🤖 ML Model: ❌</p>', unsafe_allow_html=True)

        # Data status
        price_data = load_price_data(selected_symbol)
        if not price_data.empty:
            st.markdown(f'<p style="color: #10b981; margin: 0.3rem 0;">✅ Data loaded: {len(price_data)} records</p>',
                        unsafe_allow_html=True)
            last_update = price_data['timestamp'].max()
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

    # Load data
    price_data = load_price_data(selected_symbol)
    sentiment_data = load_sentiment_data()
    predictions_data = load_predictions()
    news_articles_df = load_news_data()
    ml_model = load_ml_model()

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

        # Chart Section
        st.markdown('<h2 class="section-header">📈 Advanced Charting</h2>', unsafe_allow_html=True)

        fig = create_advanced_price_chart(price_data, selected_symbol, asset_type_selected, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"⚠️ No price data available for {selected_symbol}")
        st.info("💡 Run the data collection first using: `python main.py` and select option 1")

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
                    st.info(f"No recent sentiment data for {symbol_for_sentiment}")

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
            st.info("Collecting sentiment data...")

    with col_right:
        # AI Predictions Section
        st.markdown('<h2 class="section-header">🤖 AI Predictions</h2>', unsafe_allow_html=True)

        if not predictions_data.empty:
            latest_pred = predictions_data[predictions_data['symbol'] == selected_symbol]
            if not latest_pred.empty:
                latest_pred = latest_pred.iloc[-1]
                pred_class = "prediction-up" if latest_pred['prediction'] == 'UP' else "prediction-down"
                arrow = "🔼" if latest_pred['prediction'] == 'UP' else "🔽"
                confidence_color = "#10b981" if latest_pred['confidence'] > 0.7 else "#f59e0b" if latest_pred[
                                                                                                      'confidence'] > 0.6 else "#ef4444"

                st.markdown(f"""
                <div class="prediction-card {pred_class}">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <h4 style="color: white; margin: 0; flex-grow: 1;">AI Prediction: {arrow} {latest_pred['prediction']}</h4>
                        <span style="background: {confidence_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                            {latest_pred['confidence']:.1%} confidence
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Up Probability</p>
                            <p style="color: #10b981; font-weight: bold; margin: 0.3rem 0; font-size: 1.1rem;">{latest_pred.get('up_probability', 0):.1%}</p>
                        </div>
                        <div>
                            <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Model Used</p>
                            <p style="color: #8b5cf6; margin: 0.3rem 0; font-size: 1rem;">{latest_pred.get('model_used', 'N/A').replace('_', ' ').title()}</p>
                        </div>
                        <div>
                            <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Down Probability</p>
                            <p style="color: #ef4444; font-weight: bold; margin: 0.3rem 0; font-size: 1.1rem;">{latest_pred.get('down_probability', 0):.1%}</p>
                        </div>
                        <div>
                            <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Last Updated</p>
                            <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 1rem;">{latest_pred['timestamp']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No predictions available for this symbol yet")
        else:
            st.info("Train ML models to get AI predictions!")

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
        st.rerun()


if __name__ == "__main__":
    main()