import time
import pandas as pd
import schedule
from datetime import datetime
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.binance_client import BinanceDataCollector
from data.reddit_client import RedditSentimentCollector
from utils.data_manager import DataManager
from config.settings import BINANCE_CONFIG, REDDIT_CONFIG, SYMBOLS, SUBREDDITS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealTimeStockSentiment:
    def __init__(self):
        self.binance_collector = BinanceDataCollector(
            BINANCE_CONFIG['api_key'],
            BINANCE_CONFIG['api_secret']
        )
        self.reddit_collector = RedditSentimentCollector(
            REDDIT_CONFIG['client_id'],
            REDDIT_CONFIG['client_secret'],
            REDDIT_CONFIG['user_agent']
        )
        self.data_manager = DataManager()
        self.is_running = False

    def collect_market_data(self):
        """Collect market data from Binance"""
        logger.info("📊 Collecting market data...")
        try:
            for symbol in SYMBOLS:
                # Get historical data
                historical_data = self.binance_collector.get_historical_data(symbol, '1h', 1)
                if not historical_data.empty:
                    self.data_manager.save_price_data(symbol, historical_data)
                    logger.info(f"✅ Collected {len(historical_data)} records for {symbol}")

                    # Show latest price
                    latest = historical_data.iloc[-1]
                    print(f"   {symbol}: ${latest['close']:,.2f} (Volume: {latest['volume']:.0f})")
                else:
                    logger.warning(f"❌ No data collected for {symbol}")

                # Small delay to avoid rate limiting
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error collecting market data: {e}")

    def collect_sentiment_data(self):
        """Collect sentiment data from Reddit"""
        logger.info("🔴 Collecting sentiment data...")
        try:
            # Use the simple method to avoid API limits
            posts_data = []
            for subreddit in SUBREDDITS:
                posts = self.reddit_collector.get_recent_posts(subreddit, limit=10)
                posts_data.extend(posts)
                time.sleep(1)  # Rate limiting

            if posts_data:
                # Create sentiment summary manually
                sentiment_summary = []
                symbol_mentions = {}

                for post in posts_data:
                    for symbol in post['symbols_mentioned']:
                        if symbol not in symbol_mentions:
                            symbol_mentions[symbol] = {
                                'total_sentiment': 0,
                                'count': 0,
                                'mentions': 0
                            }
                        symbol_mentions[symbol]['total_sentiment'] += post['sentiment']
                        symbol_mentions[symbol]['count'] += 1
                        symbol_mentions[symbol]['mentions'] += 1

                for symbol, data in symbol_mentions.items():
                    if data['count'] > 0:
                        avg_sentiment = data['total_sentiment'] / data['count']
                        sentiment_summary.append({
                            'symbol': symbol,
                            'avg_sentiment': avg_sentiment,
                            'total_mentions': data['mentions'],
                            'timestamp': datetime.now()
                        })

                df_sentiment = pd.DataFrame(sentiment_summary)
                self.data_manager.save_sentiment_data(df_sentiment)

                print("📊 Current Sentiment Summary:")
                if not df_sentiment.empty:
                    for _, row in df_sentiment.iterrows():
                        sentiment_emoji = "😊" if row['avg_sentiment'] > 0.1 else "😐" if row[
                                                                                            'avg_sentiment'] > -0.1 else "😞"
                        print(
                            f"   {row['symbol']}: {row['avg_sentiment']:.2f} {sentiment_emoji} ({row['total_mentions']} mentions)")
                else:
                    print("   No significant sentiment data found")
            else:
                logger.warning("❌ No posts data collected")

        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")

    def run_once(self):
        """Run one complete data collection cycle"""
        logger.info("🚀 Starting data collection cycle...")
        print("\n" + "=" * 50)
        print(f"🕒 Cycle started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.collect_market_data()
        self.collect_sentiment_data()

        print(f"✅ Cycle completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

    def start_scheduled(self, interval_minutes=5):
        """Start scheduled data collection"""
        logger.info(f"🔄 Starting scheduled data collection every {interval_minutes} minutes")
        self.is_running = True

        # Run immediately
        self.run_once()

        # Schedule subsequent runs
        schedule.every(interval_minutes).minutes.do(self.run_once)

        print(f"🎯 Scheduled collection started. Running every {interval_minutes} minutes.")
        print("Press Ctrl+C to stop...")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Data collection stopped by user")
            self.stop()

    def stop(self):
        """Stop data collection"""
        self.is_running = False
        logger.info("🛑 Stopping data collection...")


def main():
    app = RealTimeStockSentiment()

    print("🤖 Real-Time Stock Sentiment Tracker")
    print("=" * 40)
    print("1. Run once")
    print("2. Start scheduled collection")
    print("3. Test APIs only")

    choice = input("\nChoose option (1-3): ").strip()

    if choice == "1":
        app.run_once()
    elif choice == "2":
        interval = input("Enter interval in minutes (default 5): ").strip()
        interval = int(interval) if interval.isdigit() else 5
        app.start_scheduled(interval)
    elif choice == "3":
        # Test APIs
        from notebooks.test_both_apis import test_binance, test_reddit
        test_binance()
        test_reddit()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()