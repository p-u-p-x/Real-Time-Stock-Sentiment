import praw
import pandas as pd
from datetime import datetime
import logging
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditSentimentCollector:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi'],
            'ETH': ['ethereum', 'eth', 'vitalik'],
            'ADA': ['cardano', 'ada', 'charles hoskinson'],
            'DOT': ['polkadot', 'dot', 'gavin wood'],
            'LINK': ['chainlink', 'link', 'oracle']
        }

    def extract_symbols_from_text(self, text):
        """Extract cryptocurrency symbols from text"""
        if not text:
            return []

        text_lower = text.lower()
        mentioned_symbols = []

        for symbol, keywords in self.crypto_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_symbols.append(symbol)

        return mentioned_symbols

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity  # -1 to 1 scale
        except:
            return 0.0

    def get_recent_posts(self, subreddit_name, limit=50):
        """Get recent posts from a subreddit"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []

            for post in subreddit.new(limit=limit):
                # Analyze post title
                title_sentiment = self.analyze_sentiment(post.title)
                mentioned_symbols = self.extract_symbols_from_text(post.title)

                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'sentiment': title_sentiment,
                    'symbols_mentioned': mentioned_symbols,
                    'source': f'r/{subreddit_name}',
                    'type': 'post'
                }
                posts_data.append(post_data)

            logger.info(f"Fetched {len(posts_data)} posts from r/{subreddit_name}")
            return posts_data

        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
            return []


if __name__ == "__main__":
    from config.settings import REDDIT_CONFIG

    collector = RedditSentimentCollector(**REDDIT_CONFIG)
    posts = collector.get_recent_posts('CryptoCurrency', limit=5)
    print(f"Fetched {len(posts)} posts")