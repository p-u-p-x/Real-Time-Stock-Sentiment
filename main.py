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
from models.price_predictor import PricePredictor
from models.feature_engineer import FeatureEngineer
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

    def collect_prediction_features(self, symbol):
        """Collect features for ML prediction"""
        try:
            # Load recent price data
            price_file = f"data/raw/{symbol}_prices.csv"
            if not os.path.exists(price_file):
                return None

            price_data = pd.read_csv(price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

            # Load sentiment data
            sentiment_data = pd.DataFrame()
            sentiment_file = "data/raw/sentiment.csv"
            if os.path.exists(sentiment_file):
                sentiment_data = pd.read_csv(sentiment_file)
                if 'timestamp' in sentiment_data.columns:
                    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])

            # Prepare features using feature engineer
            if len(price_data) >= 24:  # Need enough data for features
                X, y, feature_cols = self.feature_engineer.prepare_training_data(price_data, sentiment_data)
                if not X.empty:
                    latest_features = X.iloc[-1].to_dict()
                    return latest_features

            return None

        except Exception as e:
            logger.error(f"Error collecting prediction features for {symbol}: {e}")
            return None

    def generate_predictions(self):
        """Generate ML predictions for all symbols"""
        if not self.predictor or not self.predictor.best_model:
            logger.warning("No ML model available for predictions")
            return {}

        predictions = {}
        logger.info("🤖 Generating ML predictions...")

        for symbol in SYMBOLS:
            features = self.collect_prediction_features(symbol)
            if features:
                prediction = self.predictor.predict_next_hour(features)
                predictions[symbol] = prediction

                # Log prediction
                logger.info(f"📊 {symbol}: {prediction['prediction']} "
                            f"(Confidence: {prediction['confidence']:.2f})")

                # Save prediction to file
                self.save_prediction(symbol, prediction)
            else:
                logger.warning(f"Could not generate features for {symbol}")

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

        # Generate ML predictions if model is available
        if self.predictor and self.predictor.best_model:
            predictions = self.generate_predictions()
            if predictions:
                print("\n🤖 ML Predictions Summary:")
                for symbol, pred in predictions.items():
                    arrow = "🔼" if pred['prediction'] == 'UP' else "🔽"
                    print(f"   {symbol}: {arrow} {pred['prediction']} ({pred['confidence']:.1%} confidence)")

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
    print("4. Train ML models")

    choice = input("\nChoose option (1-4): ").strip()

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
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()