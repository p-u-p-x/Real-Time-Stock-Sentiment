import praw
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditSentimentCollectorMinimal:
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

    def simple_sentiment(self, text):
        """Simple sentiment analysis without external libraries"""
        positive_words = ['bull', 'moon', 'rocket', 'buy', 'good', 'great', 'amazing', 'profit']
        negative_words = ['bear', 'crash', 'sell', 'bad', 'worried', 'falling', 'loss']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def get_recent_posts(self, subreddit_name, limit=50):
        """Get recent posts from a subreddit"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []

            for post in subreddit.new(limit=limit):
                # Use simple sentiment analysis
                sentiment = self.simple_sentiment(post.title)
                mentioned_symbols = self.extract_symbols_from_text(post.title)

                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'sentiment': sentiment,
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