import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.reddit_client import RedditSentimentCollector
from config.settings import REDDIT_CONFIG


def main():
    print("🔴 Testing Reddit Connection...")

    try:
        # Initialize Reddit collector
        collector = RedditSentimentCollector(
            REDDIT_CONFIG['client_id'],
            REDDIT_CONFIG['client_secret'],
            REDDIT_CONFIG['user_agent']
        )

        print("✅ Reddit collector initialized successfully!")

        # Test sentiment analysis
        print("\n📊 Testing sentiment analysis...")
        test_texts = [
            "Bitcoin is going to the moon! 🚀",
            "I'm worried about Ethereum prices falling.",
            "ADA looks stable today."
        ]

        for text in test_texts:
            sentiment = collector.analyze_sentiment(text)
            symbols = collector.extract_symbols_from_text(text)
            print(f"Text: '{text}'")
            print(f"Sentiment: {sentiment:.2f}, Symbols: {symbols}")
            print()

        # Test fetching posts
        print("🔄 Fetching recent posts from r/CryptoCurrency...")
        posts = collector.get_recent_posts('CryptoCurrency', limit=3)

        if posts:
            print(f"✅ Successfully fetched {len(posts)} items!")
            for i, post in enumerate(posts[:3]):  # Show first 3
                print(f"{i + 1}. {post['title'][:60]}... (Sentiment: {post['sentiment']:.2f})")
        else:
            print("❌ No posts fetched")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    main()