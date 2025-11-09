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

        # Enhanced keyword mapping for both crypto and stocks
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi'],
            'ETH': ['ethereum', 'eth', 'vitalik'],
            'ADA': ['cardano', 'ada', 'charles hoskinson'],
            'DOT': ['polkadot', 'dot', 'gavin wood'],
            'LINK': ['chainlink', 'link', 'oracle'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'DOGE': ['dogecoin', 'doge']
        }

        self.stock_keywords = {
            'AAPL': ['apple', 'aapl', 'iphone', 'ipad', 'macbook'],
            'TSLA': ['tesla', 'tsla', 'elon musk', 'model s', 'model 3', 'model x', 'model y'],
            'MSFT': ['microsoft', 'msft', 'windows', 'azure', 'surface', 'satya nadella'],
            'AMZN': ['amazon', 'amzn', 'jeff bezos', 'aws', 'prime'],
            'GOOGL': ['google', 'alphabet', 'googl', 'sundar pichai', 'android', 'youtube'],
            'NVDA': ['nvidia', 'nvda', 'jensen huang', 'gpu', 'rtx'],
            'META': ['meta', 'facebook', 'fb', 'mark zuckerberg', 'instagram', 'whatsapp'],
            'JPM': ['jpmorgan', 'jpm', 'chase', 'jamie dimon'],
            'NFLX': ['netflix', 'nflx', 'reed hastings'],
            'AMD': ['amd', 'advanced micro devices', 'lisa su']
        }

    def extract_symbols_from_text(self, text):
        """Extract cryptocurrency and stock symbols from text"""
        if not text:
            return []

        text_lower = text.lower()
        mentioned_symbols = []

        # Check for crypto symbols
        for symbol, keywords in self.crypto_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_symbols.append(symbol)

        # Check for stock symbols
        for symbol, keywords in self.stock_keywords.items():
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

    def get_comments_for_post(self, post_id, limit=20):
        """Get comments for a specific post"""
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)

            comments_data = []
            for comment in submission.comments.list()[:limit]:
                comment_sentiment = self.analyze_sentiment(comment.body)
                mentioned_symbols = self.extract_symbols_from_text(comment.body)

                comment_data = {
                    'id': comment.id,
                    'body': comment.body[:500],  # Limit length
                    'score': comment.score,
                    'sentiment': comment_sentiment,
                    'symbols_mentioned': mentioned_symbols,
                    'type': 'comment'
                }
                comments_data.append(comment_data)

            return comments_data
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            return []