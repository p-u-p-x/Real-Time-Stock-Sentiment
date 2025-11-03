import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self):
        self.data_dir = "data"
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories safely"""
        try:
            os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
            os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            logger.info("✅ Directories created/verified successfully")
        except Exception as e:
            logger.warning(f"Directory creation warning (may already exist): {e}")

    def save_price_data(self, symbol, data):
        """Save price data to CSV"""
        try:
            filename = f"{self.data_dir}/raw/{symbol}_prices.csv"

            # If file exists, append new data; otherwise create new file
            if os.path.exists(filename):
                existing_data = pd.read_csv(filename)
                # Combine and remove duplicates
                combined_data = pd.concat([existing_data, data]).drop_duplicates(subset=['timestamp'], keep='last')
                combined_data.to_csv(filename, index=False)
                logger.info(f"✅ Updated price data for {symbol} (now {len(combined_data)} records)")
            else:
                data.to_csv(filename, index=False)
                logger.info(f"✅ Created new price data file for {symbol} with {len(data)} records")

        except Exception as e:
            logger.error(f"❌ Error saving price data for {symbol}: {e}")

    def save_sentiment_data(self, sentiment_data):
        """Save sentiment data to CSV"""
        try:
            if sentiment_data.empty:
                logger.warning("No sentiment data to save")
                return

            filename = f"{self.data_dir}/raw/sentiment.csv"

            # If file exists, append new data; otherwise create new file
            if os.path.exists(filename):
                existing_data = pd.read_csv(filename)
                combined_data = pd.concat([existing_data, sentiment_data]).drop_duplicates()
                combined_data.to_csv(filename, index=False)
                logger.info(f"✅ Updated sentiment data (now {len(combined_data)} records)")
            else:
                sentiment_data.to_csv(filename, index=False)
                logger.info(f"✅ Created new sentiment data file with {len(sentiment_data)} records")

        except Exception as e:
            logger.error(f"❌ Error saving sentiment data: {e}")

    def load_recent_data(self, symbol, hours=24):
        """Load recent data for a symbol"""
        filename = f"{self.data_dir}/raw/{symbol}_prices.csv"
        try:
            if os.path.exists(filename):
                data = pd.read_csv(filename)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_data = data[data['timestamp'] >= cutoff_time]
                logger.info(f"✅ Loaded {len(recent_data)} recent records for {symbol}")
                return recent_data
            else:
                logger.warning(f"❌ No data file found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            return pd.DataFrame()