import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.models.feature_engineer import FeatureEngineer
from src.models.price_predictor import PricePredictor
from config.settings import SYMBOLS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data():
    """Load all available data for training"""
    all_price_data = []
    all_sentiment_data = []

    # Load price data for all symbols
    for symbol in SYMBOLS:
        price_file = f"data/raw/{symbol}_prices.csv"
        if os.path.exists(price_file):
            df = pd.read_csv(price_file)
            df['symbol'] = symbol  # Add symbol column for identification
            all_price_data.append(df)
            logger.info(f"Loaded {len(df)} records for {symbol}")
        else:
            logger.warning(f"No data file found for {symbol}")

    # Load sentiment data
    sentiment_file = "data/raw/sentiment.csv"
    if os.path.exists(sentiment_file):
        sentiment_data = pd.read_csv(sentiment_file)
        logger.info(f"Loaded {len(sentiment_data)} sentiment records")
    else:
        sentiment_data = pd.DataFrame()
        logger.warning("No sentiment data found")

    if all_price_data:
        combined_price_data = pd.concat(all_price_data, ignore_index=True)
        return combined_price_data, sentiment_data
    else:
        raise ValueError("No price data available for training")


def validate_features(X, y):
    """Validate that features are ready for training"""
    if X.empty or y.empty:
        return False

    # Check for NaN values
    if X.isna().any().any() or y.isna().any():
        logger.warning("NaN values found in features or target")
        return False

    # Check that all features are numeric
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.error(f"Non-numeric columns found: {list(non_numeric_cols)}")
        return False

    # Check target distribution
    target_counts = y.value_counts()
    if len(target_counts) < 2:
        logger.error("Target has only one class")
        return False

    logger.info(f"Target distribution: {target_counts.to_dict()}")
    return True


def main():
    """Main training function"""
    logger.info("🚀 Starting ML Model Training...")

    try:
        # Load data
        price_data, sentiment_data = load_training_data()
        logger.info(f"Total training records: {len(price_data)}")

        if len(price_data) < 50:
            logger.error("Insufficient data for training. Need at least 50 records.")
            return

        # Feature engineering
        feature_engineer = FeatureEngineer()
        X, y, feature_columns = feature_engineer.prepare_training_data(price_data, sentiment_data)

        # Validate features
        if not validate_features(X, y):
            logger.error("Feature validation failed")
            return

        logger.info(f"Features: {len(feature_columns)}")
        logger.info(f"Feature names: {feature_columns}")

        # Train models
        predictor = PricePredictor()
        performance = predictor.train_models(X, y)

        if performance:
            # Save models
            predictor.save_model('models/trained_models.pkl')

            # Print results
            logger.info("🎯 Training Completed!")
            logger.info(f"Best Model: {performance.get('best_model', 'N/A')}")
            logger.info(f"Accuracy: {performance.get('accuracy', 0):.3f}")
            logger.info(f"CV Score: {performance.get('cv_score', 0):.3f}")

            # Test prediction with latest data
            if not X.empty:
                latest_features = X.iloc[-1].to_dict()
                prediction = predictor.predict_next_hour(latest_features)
                logger.info(f"Sample Prediction: {prediction}")
        else:
            logger.error("Model training failed")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()