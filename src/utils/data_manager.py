import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self):
        self.data_dir = "data"
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.create_directories()

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def save_price_data(self, symbol, df):
        """Save price data to CSV"""
        try:
            filename = os.path.join(self.raw_dir, f"{symbol}_prices.csv")

            if os.path.exists(filename):
                # Append to existing file
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Remove duplicates based on timestamp
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                combined_df.to_csv(filename, index=False)
                logger.info(f"✅ Updated price data for {symbol} (now {len(combined_df)} records)")
            else:
                # Create new file
                df.to_csv(filename, index=False)
                logger.info(f"✅ Created new price data file for {symbol} with {len(df)} records")

        except Exception as e:
            logger.error(f"Error saving price data for {symbol}: {e}")

    def save_sentiment_data(self, df):
        """Save sentiment data to CSV"""
        try:
            filename = os.path.join(self.raw_dir, "sentiment.csv")

            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
                combined_df.to_csv(filename, index=False)
                logger.info(f"✅ Updated sentiment data (now {len(combined_df)} records)")
            else:
                df.to_csv(filename, index=False)
                logger.info(f"✅ Created new sentiment data with {len(df)} records")

        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")

    def save_news_data(self, sentiment_data, articles_data):
        """Save news sentiment and articles data"""
        try:
            # Save sentiment data
            if sentiment_data:
                sentiment_df = pd.DataFrame(sentiment_data)
                news_sentiment_file = os.path.join(self.raw_dir, "news_sentiment.csv")

                if os.path.exists(news_sentiment_file):
                    existing_df = pd.read_csv(news_sentiment_file)
                    combined_df = pd.concat([existing_df, sentiment_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
                    combined_df.to_csv(news_sentiment_file, index=False)
                else:
                    sentiment_df.to_csv(news_sentiment_file, index=False)

                logger.info(f"Saved news sentiment data for {len(sentiment_data)} symbols")

            # Save articles data
            if articles_data:
                articles_df = pd.DataFrame(articles_data)
                articles_file = os.path.join(self.raw_dir, "news_articles.csv")

                if os.path.exists(articles_file):
                    existing_articles = pd.read_csv(articles_file)
                    combined_articles = pd.concat([existing_articles, articles_df], ignore_index=True)
                    combined_articles = combined_articles.drop_duplicates(subset=['title'], keep='last')
                    combined_articles.to_csv(articles_file, index=False)
                else:
                    articles_df.to_csv(articles_file, index=False)

                logger.info(f"Saved {len(articles_data)} news articles")

        except Exception as e:
            logger.error(f"Error saving news data: {e}")

    def load_sentiment_data(self):
        """Load combined sentiment data from both Reddit and News"""
        try:
            sentiment_file = os.path.join(self.raw_dir, "sentiment.csv")
            news_sentiment_file = os.path.join(self.raw_dir, "news_sentiment.csv")

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
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()

    def load_news_sentiment(self):
        """Load news sentiment data specifically"""
        try:
            news_sentiment_file = os.path.join(self.raw_dir, "news_sentiment.csv")
            if os.path.exists(news_sentiment_file):
                df = pd.read_csv(news_sentiment_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading news sentiment: {e}")
            return pd.DataFrame()

    def load_news_articles(self, limit=50):
        """Load recent news articles"""
        try:
            articles_file = os.path.join(self.raw_dir, "news_articles.csv")
            if os.path.exists(articles_file):
                df = pd.read_csv(articles_file)
                if 'published_at' in df.columns:
                    df['published_at'] = pd.to_datetime(df['published_at'])
                    df = df.sort_values('published_at', ascending=False)
                return df.head(limit)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading news articles: {e}")
            return pd.DataFrame()

    def load_price_data(self, symbol):
        """Load price data for a symbol"""
        try:
            filename = os.path.join(self.raw_dir, f"{symbol}_prices.csv")
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading price data for {symbol}: {e}")
            return pd.DataFrame()