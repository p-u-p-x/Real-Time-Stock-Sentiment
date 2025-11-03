"""
Real-Time Crypto Sentiment Tracker
Streamlit Dashboard Entry Point
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Crypto Sentiment Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .positive { color: #00ff00; }
    .negative { color: #ff0000; }
    .neutral { color: #ffaa00; }
</style>
""", unsafe_allow_html=True)

# Default symbols if config is not available
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']


def load_price_data(symbol):
    """Load price data for a symbol"""
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
    """Load sentiment data"""
    try:
        filename = "data/raw/sentiment.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return pd.DataFrame()


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Real-Time Crypto Sentiment Tracker</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Controls")
    selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", DEFAULT_SYMBOLS)

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30 seconds)", value=False)
    if auto_refresh:
        st.rerun()

    # Main content
    col1, col2, col3 = st.columns(3)

    # Load data
    price_data = load_price_data(selected_symbol)
    sentiment_data = load_sentiment_data()

    if not price_data.empty:
        # Latest price metrics
        latest_price = price_data.iloc[-1]
        prev_price = price_data.iloc[-2] if len(price_data) > 1 else latest_price

        price_change = latest_price['close'] - prev_price['close']
        price_change_pct = (price_change / prev_price['close']) * 100

        with col1:
            st.metric(
                label=f"{selected_symbol} Price",
                value=f"${latest_price['close']:,.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )

        with col2:
            st.metric(
                label="Volume",
                value=f"{latest_price['volume']:,.0f}",
            )

        with col3:
            rsi_value = latest_price.get('rsi', 50)  # Default if RSI not available
            rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
            st.metric(
                label="RSI",
                value=f"{rsi_value:.1f}",
                delta=rsi_status
            )

        # Price chart
        st.subheader(f"📈 {selected_symbol} Price Chart")

        # Create line chart
        fig_price = px.line(
            price_data,
            x='timestamp',
            y='close',
            title=f"{selected_symbol} Price Movement"
        )

        fig_price.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            template="plotly_white"
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # Technical indicators
        col4, col5, col6 = st.columns(3)

        with col4:
            st.subheader("📊 Price Analysis")
            st.write(f"**High:** ${price_data['high'].max():,.2f}")
            st.write(f"**Low:** ${price_data['low'].min():,.2f}")
            st.write(f"**Volume:** {latest_price['volume']:,.0f}")

        with col5:
            st.subheader("🔧 Technicals")
            st.write(f"**RSI:** {rsi_value:.1f}")
            st.write(f"**Change:** {price_change_pct:+.2f}%")
            st.write(f"**Data Points:** {len(price_data)}")

    else:
        st.warning(f"⚠️ No price data available for {selected_symbol}")
        st.info("""
        💡 **To get data:**
        1. Run `python main.py` to collect data
        2. Select option 1 to run once
        3. Refresh this page
        """)

    # Sentiment Analysis Section
    st.subheader("🔴 Social Sentiment Analysis")

    if not sentiment_data.empty:
        # Current sentiment for selected symbol
        symbol_name = selected_symbol.replace('USDT', '')
        symbol_sentiment = sentiment_data[
            sentiment_data['symbol'] == symbol_name
            ]

        if not symbol_sentiment.empty:
            latest_sentiment = symbol_sentiment.iloc[-1]
            sentiment_value = latest_sentiment.get('avg_sentiment', 0)

            # Sentiment gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{selected_symbol} Sentiment"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "lightcoral"},
                        {'range': [-0.3, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 1], 'color': "lightgreen"}
                    ]
                }
            ))

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Sentiment interpretation
            if sentiment_value > 0.3:
                st.success("🎉 Strong Positive Sentiment - Bullish")
            elif sentiment_value > 0.1:
                st.info("👍 Mildly Positive - Cautiously optimistic")
            elif sentiment_value > -0.1:
                st.warning("🤔 Neutral - Market uncertainty")
            elif sentiment_value > -0.3:
                st.error("👎 Mildly Negative - Caution advised")
            else:
                st.error("💀 Strong Negative - Bearish")

        else:
            st.info(f"ℹ️ No sentiment data for {selected_symbol}")

        # Most mentioned cryptocurrencies
        st.subheader("🏆 Most Discussed Cryptos")

        if 'symbol' in sentiment_data.columns:
            if 'total_mentions' in sentiment_data.columns:
                mention_counts = sentiment_data.groupby('symbol')['total_mentions'].sum().sort_values(ascending=False)
            else:
                # Count occurrences if total_mentions not available
                mention_counts = sentiment_data['symbol'].value_counts()

            col7, col8 = st.columns(2)

            with col7:
                st.write("**Mention Ranking:**")
                for i, (symbol, mentions) in enumerate(mention_counts.head(5).items(), 1):
                    st.write(f"{i}. {symbol}: {mentions} mentions")

    else:
        st.info("""
        ℹ️ **No sentiment data yet**

        To collect sentiment data:
        1. Run `python main.py` 
        2. Select option 1
        3. Wait for Reddit data collection
        4. Refresh this page
        """)

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 Instructions")
    st.sidebar.write("""
    1. Run `python main.py` to collect data
    2. Select cryptocurrencies to monitor
    3. View real-time prices & sentiment
    4. Auto-refresh for live updates
    """)


if __name__ == "__main__":
    main()