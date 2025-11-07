import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

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

# Enhanced Professional CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #1e3a8a;
        --secondary: #3b82f6;
        --accent: #06d6a0;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
        --sidebar-bg: #0f172a;
        --header-bg: #1e293b;
    }

    .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1e293b 100%);
        color: var(--text-primary);
    }

    /* Header and sidebar styling */
    header[data-testid="stHeader"] {
        background: var(--header-bg) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-bg) 0%, #1e293b 100%) !important;
    }

    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .metric-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    .prediction-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 1rem 0;
    }

    .prediction-up { border-left-color: #06d6a0; }
    .prediction-down { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)


def load_price_data(symbol):
    """Load price data for any symbol"""
    try:
        filename = f"data/raw/{symbol}_prices.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()


def load_sentiment_data():
    """Load combined sentiment data"""
    try:
        sentiment_file = "data/raw/sentiment.csv"
        news_sentiment_file = "data/raw/news_sentiment.csv"

        sentiment_data = pd.DataFrame()
        news_sentiment = pd.DataFrame()

        if os.path.exists(sentiment_file):
            sentiment_data = pd.read_csv(sentiment_file)
            if 'timestamp' in sentiment_data.columns:
                sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])

        if os.path.exists(news_sentiment_file):
            news_sentiment = pd.read_csv(news_sentiment_file)
            if 'timestamp' in news_sentiment.columns:
                news_sentiment['timestamp'] = pd.to_datetime(news_sentiment['timestamp'])

        # Combine both sentiment sources
        combined_sentiment = pd.concat([sentiment_data, news_sentiment], ignore_index=True)
        return combined_sentiment

    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
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


def create_price_chart(df, symbol, asset_type):
    """Create price chart for stocks or crypto"""
    if df.empty:
        return None

    fig = go.Figure()

    if asset_type == "crypto":
        # Candlestick for crypto
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#06d6a0',
            decreasing_line_color='#ef4444'
        ))
    else:
        # Line chart for stocks
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#3b82f6', width=2)
        ))

    fig.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        title=dict(
            text=f"{symbol} Price Movement",
            font=dict(color='#3b82f6', size=20)
        ),
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155'),
        showlegend=False
    )

    return fig


def create_sentiment_gauge(sentiment_value, symbol, source):
    """Create sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{symbol} {source} Sentiment", 'font': {'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [-1, 1], 'tickcolor': '#f8fafc'},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(148, 163, 184, 0.3)'},
                {'range': [0.3, 1], 'color': 'rgba(6, 214, 160, 0.3)'}
            ]
        }
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc"}
    )

    return fig


def main():
    # Header Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">Quantum Trader AI</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">Advanced Stock & Crypto Analytics Platform</p>',
            unsafe_allow_html=True)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown(
            '<div style="color: #3b82f6; font-weight: bold; font-size: 1.2rem; text-align: center; margin-bottom: 2rem;">🔧 Dashboard Controls</div>',
            unsafe_allow_html=True)

        # Asset type selection
        asset_type = st.selectbox(
            "Select Asset Type:",
            ["All Assets", "Cryptocurrencies", "Stocks"],
            index=0
        )

        # Symbol selection based on asset type
        if asset_type == "Stocks":
            available_symbols = STOCK_SYMBOLS
        elif asset_type == "Cryptocurrencies":
            available_symbols = CRYPTO_SYMBOLS
        else:
            available_symbols = ALL_SYMBOLS

        symbol_names = [f"{sym} - {ASSET_DISPLAY_NAMES.get(sym, sym)}" for sym in available_symbols]
        selected_symbol = st.selectbox("Select Asset:", symbol_names)
        selected_symbol = selected_symbol.split(' - ')[0]

        # Determine asset type for selected symbol
        asset_type_selected = "crypto" if selected_symbol in CRYPTO_SYMBOLS else "stock"

        # Time range
        time_range = st.selectbox("Time Range:", ["24H", "7D", "1M", "3M"], index=0)

        # Auto-refresh
        auto_refresh = st.checkbox("Enable Auto-refresh", value=False)

        st.markdown("---")

        # Data status
        price_data = load_price_data(selected_symbol)
        if not price_data.empty:
            st.success(f"✅ Data loaded: {len(price_data)} records")
            last_update = price_data['timestamp'].max()
            st.info(f"📅 Last update: {last_update}")
        else:
            st.warning("⚠️ No data available")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # ===== MAIN DASHBOARD =====

    # Load data
    price_data = load_price_data(selected_symbol)
    sentiment_data = load_sentiment_data()
    predictions_data = load_predictions()

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
                <p style="color: #06d6a0; font-size: 1.8rem; font-weight: bold; margin: 0;">${latest_price['close']:,.2f}</p>
                <p style="color: {'#06d6a0' if price_change_pct >= 0 else '#ef4444'}; margin: 0.5rem 0 0 0;">
                    {price_change_pct:+.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">24H Volume</p>
                <p style="color: #3b82f6; font-size: 1.8rem; font-weight: bold; margin: 0;">{latest_price['volume']:,.0f}</p>
                <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Market Activity</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            rsi_value = latest_price.get('rsi', 50)
            rsi_color = '#06d6a0' if 30 <= rsi_value <= 70 else '#ef4444'
            rsi_status = "Optimal" if 30 <= rsi_value <= 70 else "Extreme"

            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">RSI Indicator</p>
                <p style="color: {rsi_color}; font-size: 1.8rem; font-weight: bold; margin: 0;">{rsi_value:.1f}</p>
                <p style="color: {rsi_color}; margin: 0.5rem 0 0 0;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            volatility = price_data['close'].pct_change().std() * 100
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #94a3b8; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Volatility</p>
                <p style="color: #f59e0b; font-size: 1.8rem; font-weight: bold; margin: 0;">{volatility:.2f}%</p>
                <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Price Stability</p>
            </div>
            """, unsafe_allow_html=True)

        # Price Chart
        st.markdown(
            '<h3 style="color: #3b82f6; border-bottom: 2px solid #334155; padding-bottom: 0.5rem;">📈 Price Analysis</h3>',
            unsafe_allow_html=True)
        fig = create_price_chart(price_data, selected_symbol, asset_type_selected)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"⚠️ No price data available for {selected_symbol}")
        st.info("💡 Run the data collection first using: `python main.py` and select option 1")

    # Sentiment Analysis Section
    st.markdown(
        '<h3 style="color: #3b82f6; border-bottom: 2px solid #334155; padding-bottom: 0.5rem;">📊 Market Sentiment</h3>',
        unsafe_allow_html=True)

    if not sentiment_data.empty:
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""
            <div class="metric-card">
                <p style="color: #3b82f6; font-weight: bold; margin-bottom: 1rem;">Social Sentiment Analysis</p>
            """, unsafe_allow_html=True)

            symbol_name = selected_symbol.replace('USDT', '')
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol_name]

            if not symbol_sentiment.empty:
                latest_sentiment = symbol_sentiment.iloc[-1]
                sentiment_value = latest_sentiment.get('avg_sentiment', 0)
                source = latest_sentiment.get('source', 'combined')

                fig_gauge = create_sentiment_gauge(sentiment_value, symbol_name, source.title())
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.info("No recent sentiment data for this symbol")

            st.markdown('</div>', unsafe_allow_html=True)

        with col6:
            st.markdown("""
            <div class="metric-card">
                <p style="color: #3b82f6; font-weight: bold; margin-bottom: 1rem;">Trending Assets</p>
            """, unsafe_allow_html=True)

            if 'symbol' in sentiment_data.columns:
                # Get assets with highest mention counts
                mention_counts = sentiment_data['symbol'].value_counts().head(8)

                for i, (symbol, mentions) in enumerate(mention_counts.items(), 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    # Get average sentiment for this symbol
                    avg_sentiment = sentiment_data[sentiment_data['symbol'] == symbol]['avg_sentiment'].mean()
                    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😞"

                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <div>
                            <span style="color: #f8fafc; font-weight: 500;">{medal} {symbol}</span>
                            <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.5rem;">{sentiment_emoji} {avg_sentiment:.2f}</span>
                        </div>
                        <span style="color: #3b82f6; font-weight: bold; background: rgba(59, 130, 246, 0.1); padding: 0.25rem 0.75rem; border-radius: 12px;">{mentions}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("ℹ️ No sentiment data collected yet. Run the data collection first.")

    # AI Predictions Section
    st.markdown(
        '<h3 style="color: #3b82f6; border-bottom: 2px solid #334155; padding-bottom: 0.5rem;">🤖 AI Predictions</h3>',
        unsafe_allow_html=True)

    if not predictions_data.empty:
        latest_pred = predictions_data[predictions_data['symbol'] == selected_symbol]
        if not latest_pred.empty:
            latest_pred = latest_pred.iloc[-1]
            pred_class = "prediction-up" if latest_pred['prediction'] == 'UP' else "prediction-down"
            arrow = "🔼" if latest_pred['prediction'] == 'UP' else "🔽"
            confidence_color = "#06d6a0" if latest_pred['confidence'] > 0.7 else "#f59e0b" if latest_pred[
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
                        <p style="color: #06d6a0; font-weight: bold; margin: 0.3rem 0; font-size: 1.1rem;">{latest_pred.get('up_probability', 0):.1%}</p>
                    </div>
                    <div>
                        <p style="color: #94a3b8; margin: 0.3rem 0; font-size: 0.9rem;">Model Used</p>
                        <p style="color: #3b82f6; margin: 0.3rem 0; font-size: 1rem;">{latest_pred.get('model_used', 'N/A').replace('_', ' ').title()}</p>
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
            st.info("No predictions available for this symbol yet.")
    else:
        st.info("Train ML models to get AI predictions with 81.2% accuracy!")

    # Market Overview Section
    st.markdown(
        '<h3 style="color: #3b82f6; border-bottom: 2px solid #334155; padding-bottom: 0.5rem;">🌐 Market Overview</h3>',
        unsafe_allow_html=True)

    # Create a quick overview of top assets
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #3b82f6; font-weight: bold; margin-bottom: 1rem;">Top Cryptocurrencies</p>
        """, unsafe_allow_html=True)

        crypto_prices = []
        for symbol in CRYPTO_SYMBOLS[:4]:  # Show top 4
            data = load_price_data(symbol)
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100

                crypto_prices.append({
                    'symbol': symbol,
                    'price': latest['close'],
                    'change': change_pct
                })

        for crypto in crypto_prices:
            change_color = "#06d6a0" if crypto['change'] >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #f8fafc; font-weight: 500;">{crypto['symbol'].replace('USDT', '')}</span>
                <div style="text-align: right;">
                    <div style="color: #f8fafc; font-weight: 500;">${crypto['price']:,.2f}</div>
                    <div style="color: {change_color}; font-size: 0.8rem;">{crypto['change']:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col8:
        st.markdown("""
        <div class="metric-card">
            <p style="color: #3b82f6; font-weight: bold; margin-bottom: 1rem;">Top Stocks</p>
        """, unsafe_allow_html=True)

        stock_prices = []
        for symbol in STOCK_SYMBOLS[:4]:  # Show top 4
            data = load_price_data(symbol)
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100

                stock_prices.append({
                    'symbol': symbol,
                    'price': latest['close'],
                    'change': change_pct
                })

        for stock in stock_prices:
            change_color = "#06d6a0" if stock['change'] >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #f8fafc; font-weight: 500;">{stock['symbol']}</span>
                <div style="text-align: right;">
                    <div style="color: #f8fafc; font-weight: 500;">${stock['price']:,.2f}</div>
                    <div style="color: {change_color}; font-size: 0.8rem;">{stock['change']:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh:
        st.rerun()


if __name__ == "__main__":
    main()