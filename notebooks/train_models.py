import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.models.feature_engineer import FeatureEngineer
from src.models.price_predictor import PricePredictor
from config.settings import ALL_SYMBOLS, CRYPTO_SYMBOLS, STOCK_SYMBOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(symbols):
    """Load and combine training data from multiple symbols"""
    all_price_data = []
    all_sentiment_data = []

    for symbol in symbols:
        try:
            # Load price data
            price_file = f"data/raw/{symbol}_prices.csv"
            if os.path.exists(price_file):
                price_data = pd.read_csv(price_file)
                price_data['symbol'] = symbol
                if 'timestamp' in price_data.columns:
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                all_price_data.append(price_data)
                logger.info(f"Loaded price data for {symbol}: {len(price_data)} records")

            # Load sentiment data
            sentiment_file = "data/raw/sentiment.csv"
            if os.path.exists(sentiment_file):
                sentiment_data = pd.read_csv(sentiment_file)
                if 'timestamp' in sentiment_data.columns:
                    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
                # Filter for current symbol
                symbol_name = symbol.replace('USDT', '') if 'USDT' in symbol else symbol
                symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol_name]
                if not symbol_sentiment.empty:
                    all_sentiment_data.append(symbol_sentiment)

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")

    # Combine all data
    combined_price_data = pd.concat(all_price_data, ignore_index=True) if all_price_data else pd.DataFrame()
    combined_sentiment_data = pd.concat(all_sentiment_data, ignore_index=True) if all_sentiment_data else pd.DataFrame()

    logger.info(f"Combined data - Price: {len(combined_price_data)}, Sentiment: {len(combined_sentiment_data)}")
    return combined_price_data, combined_sentiment_data


def prepare_features(price_data, sentiment_data):
    """Prepare features for training"""
    try:
        feature_engineer = FeatureEngineer()

        if price_data.empty:
            logger.error("No price data available for feature engineering")
            return pd.DataFrame(), pd.Series(), []

        # Prepare training data
        X, y, feature_columns = feature_engineer.prepare_training_data(price_data, sentiment_data)

        if X.empty or y.empty:
            logger.error("No features or target generated")
            return pd.DataFrame(), pd.Series(), []

        logger.info(f"Feature engineering completed - X: {X.shape}, y: {y.shape}")
        logger.info(f"Feature columns: {len(feature_columns)}")

        return X, y, feature_columns

    except Exception as e:
        logger.error(f"Error in feature preparation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.Series(), []


def train_and_evaluate_models(X, y):
    """Train and evaluate machine learning models with error handling"""
    try:
        if X.empty or y.empty:
            logger.error("No data available for training")
            return None

        # Initialize and train predictor
        predictor = PricePredictor()

        logger.info("Starting model training...")
        performance = predictor.train_models(X, y)

        if performance and predictor.best_model:
            logger.info("✅ Model training completed successfully!")

            # Use get() method to safely access performance metrics
            best_model = performance.get('best_model', 'N/A')
            accuracy = performance.get('accuracy', 0)
            f1_score = performance.get('f1_score', 0)
            roc_auc = performance.get('roc_auc', 0)

            logger.info(f"Best model: {best_model}")
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"F1 Score: {f1_score:.3f}")
            logger.info(f"ROC AUC: {roc_auc:.3f}")

            # Save the trained model
            predictor.save_model()
            return predictor
        else:
            logger.error("❌ Model training failed")
            return None

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def analyze_feature_importance(predictor, feature_columns):
    """Analyze and display feature importance"""
    try:
        if not predictor or not hasattr(predictor, 'feature_importance'):
            return

        feature_importance = predictor.feature_importance
        if not feature_importance:
            return

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)

        print("\n" + "=" * 50)
        print("TOP 20 FEATURE IMPORTANCES")
        print("=" * 50)

        for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:30} {row['importance']:.4f}")

        print("=" * 50)

    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")


def save_sample_predictions(predictor, price_data, feature_columns):
    """Save sample predictions for dashboard display"""
    try:
        if not predictor or price_data.empty:
            return

        # Create sample predictions for recent data
        recent_data = price_data.tail(100)  # Last 100 records

        predictions = []
        for _, row in recent_data.iterrows():
            try:
                # Create feature vector (simplified - in practice, use FeatureEngineer)
                features = {}
                for col in feature_columns:
                    if col in row:
                        features[col] = row[col]
                    else:
                        features[col] = 0

                # Make prediction
                prediction = predictor.predict_next_hour(features)
                prediction['symbol'] = row.get('symbol', 'UNKNOWN')
                prediction['timestamp'] = row.get('timestamp', datetime.now())
                predictions.append(prediction)

            except Exception as e:
                logger.warning(f"Error making prediction for row: {e}")
                continue

        if predictions:
            # Save to CSV for dashboard
            pred_df = pd.DataFrame(predictions)
            os.makedirs('data/processed', exist_ok=True)
            pred_df.to_csv('data/processed/predictions.csv', index=False)
            logger.info(f"Saved {len(predictions)} sample predictions")

    except Exception as e:
        logger.error(f"Error saving sample predictions: {e}")


def main():
    """Main training function"""
    print("🚀 Starting ML Model Training...")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists("data/raw"):
        print("❌ No data found. Please run data collection first.")
        print("💡 Run: python main.py and choose option 1")
        return

    # Load training data
    print("📊 Loading training data...")
    price_data, sentiment_data = load_training_data(ALL_SYMBOLS)

    if price_data.empty:
        print("❌ No price data available for training.")
        print("💡 Please run data collection first.")
        return

    # Prepare features
    print("🔧 Engineering features...")
    X, y, feature_columns = prepare_features(price_data, sentiment_data)

    if X.empty:
        print("❌ No features could be created from the data.")
        return

    # Train models
    print("🤖 Training machine learning models...")
    predictor = train_and_evaluate_models(X, y)

    if predictor:
        # Analyze feature importance
        analyze_feature_importance(predictor, feature_columns)

        print("\n🎉 Training completed successfully!")
        print(f"📈 Best Model: {predictor.best_model}")
        print(f"🎯 Accuracy: {predictor.model_performance.get('accuracy', 0):.1%}")
        print(f"📊 F1 Score: {predictor.model_performance.get('f1_score', 0):.1%}")
        print(f"🏆 ROC AUC: {predictor.model_performance.get('roc_auc', 0):.1%}")

        # Save predictions for dashboard
        save_sample_predictions(predictor, price_data, feature_columns)

    else:
        print("❌ Training failed. Check the logs for details.")


if __name__ == "__main__":
    main()