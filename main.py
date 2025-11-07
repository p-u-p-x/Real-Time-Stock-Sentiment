import time
import pandas as pd
import schedule
from datetime import datetime
import logging
import sys
import os

# Add src to path - FIXED PATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from correct modules
try:
    from src.data.binance_client import BinanceDataCollector
    from src.data.stock_client import StockDataCollector
    from src.data.reddit_client import RedditSentimentCollector
    from src.data.news_client import NewsSentimentCollector
    from src.utils.data_manager import DataManager
    from src.models.price_predictor import PricePredictor
    from src.models.feature_engineer import FeatureEngineer
    from config.settings import (BINANCE_CONFIG, REDDIT_CONFIG, NEWS_API_CONFIG,
                               CRYPTO_SYMBOLS, STOCK_SYMBOLS, ALL_SYMBOLS, SUBREDDITS)
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import structure...")
    # Alternative imports
    from data.binance_client import BinanceDataCollector
    from data.stock_client import StockDataCollector
    from data.reddit_client import RedditSentimentCollector
    from data.news_client import NewsSentimentCollector
    from utils.data_manager import DataManager
    from models.price_predictor import PricePredictor
    from models.feature_engineer import FeatureEngineer
    from config.settings import (BINANCE_CONFIG, REDDIT_CONFIG, NEWS_API_CONFIG,
                               CRYPTO_SYMBOLS, STOCK_SYMBOLS, ALL_SYMBOLS, SUBREDDITS)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealTimeMarketSentiment:
    def __init__(self):
        # Initialize data collectors
        self.binance_collector = BinanceDataCollector(
            BINANCE_CONFIG['api_key'],
            BINANCE_CONFIG['api_secret']
        )
        self.stock_collector = StockDataCollector()
        self.reddit_collector = RedditSentimentCollector(
            REDDIT_CONFIG['client_id'],
            REDDIT_CONFIG['client_secret'],
            REDDIT_CONFIG['user_agent']
        )
        self.news_collector = NewsSentimentCollector(
            NEWS_API_CONFIG['api_key'],
            NEWS_API_CONFIG['provider']
        )

        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.predictor = None
        self.is_running = False
        self.setup_prediction_engine()

    def setup_prediction_engine(self):
        """Setup ML prediction engine"""
        try:
            self.predictor = PricePredictor()
            model_path = 'models/trained_models.pkl'
            if os.path.exists(model_path):
                self.predictor.load_model(model_path)
                logger.info("✅ ML Prediction engine loaded")
                return True
            else:
                logger.warning("❌ No trained ML model found. Run training first.")
                return False
        except Exception as e:
            logger.error(f"Error setting up prediction engine: {e}")
            return False

    def collect_crypto_data(self):
        """Collect cryptocurrency data from Binance"""
        logger.info("💰 Collecting cryptocurrency data...")
        try:
            for symbol in CRYPTO_SYMBOLS:
                historical_data = self.binance_collector.get_historical_data(symbol, '1h', 1)
                if not historical_data.empty:
                    self.data_manager.save_price_data(symbol, historical_data)
                    logger.info(f"✅ Collected {len(historical_data)} records for {symbol}")

                    # Show latest price
                    latest = historical_data.iloc[-1]
                    print(f"   {symbol}: ${latest['close']:,.2f} (Volume: {latest['volume']:.0f})")
                else:
                    logger.warning(f"❌ No data collected for {symbol}")

                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.error(f"Error collecting crypto data: {e}")

    # In your main.py, update the collect_stock_data method:

    # In your main.py, update the collect_stock_data method:

    def collect_stock_data(self):
        """Collect stock data using yfinance"""
        logger.info("📈 Collecting stock data...")
        try:
            from src.data.stock_client import get_stock_data_ultra_simple

            successful_stocks = 0

            for symbol in STOCK_SYMBOLS:
                print(f"   Fetching {symbol}...")

                # Use the ultra-simple method
                historical_data = get_stock_data_ultra_simple(symbol, 7)

                if not historical_data.empty:
                    self.data_manager.save_price_data(symbol, historical_data)
                    latest = historical_data.iloc[-1]
                    print(f"   ✅ {symbol}: ${latest['close']:,.2f} (RSI: {latest.get('rsi', 0):.1f})")
                    successful_stocks += 1
                else:
                    print(f"   ❌ No data collected for {symbol}")

                time.sleep(2)  # Increased rate limiting for yfinance

            print(f"📊 Stock Collection Summary: {successful_stocks}/{len(STOCK_SYMBOLS)} successful")

        except Exception as e:
            logger.error(f"Error collecting stock data: {str(e)}")

    def collect_reddit_sentiment(self):
        """Collect sentiment data from Reddit"""
        logger.info("🔴 Collecting Reddit sentiment data...")
        try:
            posts_data = []
            for subreddit in SUBREDDITS:
                posts = self.reddit_collector.get_recent_posts(subreddit, limit=15)
                posts_data.extend(posts)
                time.sleep(1)  # Rate limiting
                print(f"   Collected {len(posts)} posts from r/{subreddit}")

            if posts_data:
                # Create sentiment summary
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
                            'timestamp': datetime.now(),
                            'source': 'reddit'
                        })

                df_sentiment = pd.DataFrame(sentiment_summary)
                self.data_manager.save_sentiment_data(df_sentiment)

                print("📊 Reddit Sentiment Summary:")
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
            logger.error(f"Error collecting Reddit sentiment: {e}")

    def collect_news_sentiment(self):
        """Collect news sentiment data"""
        logger.info("📰 Collecting news sentiment data...")
        try:
            sentiment_summary, articles = self.news_collector.get_crypto_news_sentiment()

            if sentiment_summary:
                self.data_manager.save_news_data(sentiment_summary, articles)

                print("📊 News Sentiment Summary:")
                for item in sentiment_summary:
                    sentiment_emoji = "😊" if item['avg_sentiment'] > 0.1 else "😐" if item[
                                                                                         'avg_sentiment'] > -0.1 else "😞"
                    print(
                        f"   {item['symbol']}: {item['avg_sentiment']:.2f} {sentiment_emoji} ({item['total_mentions']} mentions)")
            else:
                logger.warning("❌ No news sentiment data collected")

        except Exception as e:
            logger.error(f"Error collecting news data: {e}")

    def generate_predictions(self):
        """Generate ML predictions for all symbols"""
        if not self.predictor or not self.predictor.best_model:
            logger.warning("No ML model available for predictions")
            return {}

        predictions = {}
        logger.info("🤖 Generating ML predictions...")

        for symbol in ALL_SYMBOLS:
            try:
                # Load price data
                price_file = f"data/raw/{symbol}_prices.csv"
                if not os.path.exists(price_file):
                    continue

                price_data = pd.read_csv(price_file)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

                # Load sentiment data
                sentiment_data = self.data_manager.load_sentiment_data()
                news_sentiment = self.data_manager.load_news_sentiment()

                # Combine sentiment data
                combined_sentiment = pd.concat([sentiment_data, news_sentiment], ignore_index=True)

                # Prepare features
                if len(price_data) >= 24:
                    X, y, feature_cols = self.feature_engineer.prepare_training_data(price_data, combined_sentiment)
                    if not X.empty:
                        latest_features = X.iloc[-1].to_dict()
                        prediction = self.predictor.predict_next_hour(latest_features)
                        predictions[symbol] = prediction

                        # Log prediction
                        logger.info(f"📊 {symbol}: {prediction['prediction']} "
                                    f"(Confidence: {prediction['confidence']:.2f})")

                        # Save prediction
                        self.save_prediction(symbol, prediction)
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")

        return predictions

    def save_prediction(self, symbol, prediction):
        """Save prediction to CSV file"""
        try:
            prediction_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'up_probability': prediction['up_probability'],
                'down_probability': prediction['down_probability'],
                'model_used': prediction['model_used']
            }

            pred_file = f"data/processed/predictions.csv"
            df = pd.DataFrame([prediction_data])

            if os.path.exists(pred_file):
                df.to_csv(pred_file, mode='a', header=False, index=False)
            else:
                os.makedirs('data/processed', exist_ok=True)
                df.to_csv(pred_file, index=False)

        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

    def run_once(self):
        """Run one complete data collection cycle"""
        logger.info("🚀 Starting data collection cycle...")
        print("\n" + "=" * 60)
        print(f"🕒 Cycle started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Collect all data types
        self.collect_crypto_data()
        self.collect_stock_data()
        self.collect_reddit_sentiment()
        self.collect_news_sentiment()

        # Generate ML predictions if model is available
        if self.predictor and self.predictor.best_model:
            predictions = self.generate_predictions()
            if predictions:
                print("\n🤖 ML Predictions Summary:")
                for symbol, pred in predictions.items():
                    arrow = "🔼" if pred['prediction'] == 'UP' else "🔽"
                    confidence_color = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.6 else "🔴"
                    print(f"   {symbol}: {arrow} {pred['prediction']} {confidence_color} ({pred['confidence']:.1%})")

        print(f"✅ Cycle completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

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
    app = RealTimeMarketSentiment()

    print("🤖 Real-Time Stock & Crypto Sentiment Tracker")
    print("=" * 50)
    print("1. Run once (Collect all data)")
    print("2. Start scheduled collection")
    print("3. Test APIs only")
    print("4. Train ML models")
    print("5. Start Dashboard")

    choice = input("\nChoose option (1-5): ").strip()

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
    elif choice == "4":
        # Train ML models
        from notebooks.train_models import main as train_main
        train_main()
    elif choice == "5":
        # Start dashboard
        print("Starting dashboard...")
        os.system("streamlit run src/dashboard/app.py")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()