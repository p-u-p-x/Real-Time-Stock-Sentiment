import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Now import from config
from config.settings import SYMBOLS

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


def load_price_data(symbol):
    """Load price data for a symbol"""
    try:
        filename = f"data/raw/{symbol}_prices.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
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
    selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", SYMBOLS)
    time_range = st.sidebar.selectbox("Time Range", ["1h", "6h", "24h", "7d"])

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (10 seconds)", value=False)
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
            rsi_status = "Overbought" if latest_price['rsi'] > 70 else "Oversold" if latest_price[
                                                                                         'rsi'] < 30 else "Neutral"
            rsi_color = "negative" if latest_price['rsi'] > 70 else "positive" if latest_price[
                                                                                      'rsi'] < 30 else "neutral"
            st.metric(
                label="RSI",
                value=f"{latest_price['rsi']:.1f}",
                delta=rsi_status
            )

        # Price chart
        st.subheader(f"📈 {selected_symbol} Price Chart")

        # Create a simple line chart instead of candlestick for now
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
            st.write(f"**24h High:** ${price_data['high'].max():,.2f}")
            st.write(f"**24h Low:** ${price_data['low'].min():,.2f}")
            st.write(f"**Avg Volume:** {price_data['volume'].mean():,.0f}")

        with col5:
            st.subheader("🔧 Technicals")
            st.write(f"**RSI:** {latest_price['rsi']:.1f}")
            st.write(f"**Price Change:** {price_change_pct:+.2f}%")
            st.write(f"**Latest Volume:** {latest_price['volume']:,.0f}")

        with col6:
            st.subheader("📅 Data Info")
            st.write(f"**Data Points:** {len(price_data)}")
            st.write(
                f"**Time Range:** {price_data['timestamp'].min().strftime('%m/%d %H:%M')} to {price_data['timestamp'].max().strftime('%m/%d %H:%M')}")

    else:
        st.warning(f"⚠️ No price data available for {selected_symbol}")
        st.info("💡 Run the data collection first using: `python main.py`")

    # Sentiment Analysis Section
    st.subheader("🔴 Social Sentiment Analysis")

    if not sentiment_data.empty:
        # Filter recent sentiment
        if 'timestamp' in sentiment_data.columns:
            recent_sentiment = sentiment_data[
                sentiment_data['timestamp'] >= (datetime.now() - timedelta(hours=6))
                ]
        else:
            recent_sentiment = sentiment_data

        if not recent_sentiment.empty:
            # Current sentiment for selected symbol
            symbol_name = selected_symbol.replace('USDT', '')
            symbol_sentiment = recent_sentiment[
                recent_sentiment['symbol'] == symbol_name
                ]

            if not symbol_sentiment.empty:
                latest_sentiment = symbol_sentiment.iloc[-1]
                sentiment_value = latest_sentiment['avg_sentiment']

                # Sentiment gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
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
                    st.success("🎉 Strong Positive Sentiment - Bullish indicators")
                elif sentiment_value > 0.1:
                    st.info("👍 Mildly Positive Sentiment - Cautiously optimistic")
                elif sentiment_value > -0.1:
                    st.warning("🤔 Neutral Sentiment - Market uncertainty")
                elif sentiment_value > -0.3:
                    st.error("👎 Mildly Negative Sentiment - Caution advised")
                else:
                    st.error("💀 Strong Negative Sentiment - Bearish indicators")

                # Show mentions if available
                if 'total_mentions' in latest_sentiment:
                    st.write(f"**Mentions:** {latest_sentiment['total_mentions']}")

            # Most mentioned cryptocurrencies
            st.subheader("🏆 Most Discussed Cryptos")

            if 'symbol' in recent_sentiment.columns and 'total_mentions' in recent_sentiment.columns:
                mention_counts = recent_sentiment.groupby('symbol')['total_mentions'].sum().sort_values(ascending=False)

                col7, col8 = st.columns(2)

                with col7:
                    st.write("**Mention Ranking:**")
                    for i, (symbol, mentions) in enumerate(mention_counts.head(5).items(), 1):
                        st.write(f"{i}. {symbol}: {mentions} mentions")

                with col8:
                    # Sentiment distribution
                    if 'avg_sentiment' in recent_sentiment.columns:
                        avg_sentiment_by_symbol = recent_sentiment.groupby('symbol')[
                            'avg_sentiment'].mean().sort_values(ascending=False)
                        st.write("**Average Sentiment:**")
                        for symbol, sentiment in avg_sentiment_by_symbol.head(5).items():
                            emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
                            st.write(f"{symbol}: {sentiment:.2f} {emoji}")

        else:
            st.info("ℹ️ No recent sentiment data available (last 6 hours)")

    else:
        st.info("ℹ️ No sentiment data collected yet. Run the data collection first using: `python main.py`")

    # Data last updated
    if not price_data.empty:
        last_update = price_data['timestamp'].max()
        st.sidebar.markdown("---")
        st.sidebar.write(f"**Data last updated:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()


if __name__ == "__main__":
    main()