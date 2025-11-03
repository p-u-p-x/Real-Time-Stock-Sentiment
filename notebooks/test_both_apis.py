import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.binance_client import BinanceDataCollector
from src.data.reddit_client import RedditSentimentCollector
from config.settings import BINANCE_CONFIG, REDDIT_CONFIG, SYMBOLS
import pandas as pd


def test_binance():
    print("🔵 Testing Binance API...")
    try:
        collector = BinanceDataCollector(
            BINANCE_CONFIG['api_key'],
            BINANCE_CONFIG['api_secret']
        )

        # Test historical data
        print("📊 Fetching BTC historical data...")
        btc_data = collector.get_historical_data('BTCUSDT', '1h', 1)

        if not btc_data.empty:
            print(f"✅ Success! Fetched {len(btc_data)} BTC records")
            print(btc_data[['timestamp', 'close', 'volume', 'rsi']].tail(3))
            return True
        else:
            print("❌ No data fetched from Binance")
            return False

    except Exception as e:
        print(f"❌ Binance Error: {e}")
        return False


def test_reddit():
    print("\n🔴 Testing Reddit API...")
    try:
        collector = RedditSentimentCollector(
            REDDIT_CONFIG['client_id'],
            REDDIT_CONFIG['client_secret'],
            REDDIT_CONFIG['user_agent']
        )

        print("📝 Testing sentiment analysis...")
        test_text = "Bitcoin is amazing and going to the moon! 🚀"
        sentiment = collector.analyze_sentiment(test_text)
        symbols = collector.extract_symbols_from_text(test_text)
        print(f"Test: '{test_text}'")
        print(f"Sentiment: {sentiment:.2f}, Symbols: {symbols}")

        print("\n🔄 Fetching Reddit posts...")
        posts = collector.get_recent_posts('CryptoCurrency', limit=3)

        if posts:
            print(f"✅ Success! Fetched {len(posts)} posts")
            for i, post in enumerate(posts[:2]):
                print(f"{i + 1}. {post['title'][:60]}...")
                print(f"   Sentiment: {post['sentiment']:.2f}, Symbols: {post['symbols_mentioned']}")
            return True
        else:
            print("❌ No posts fetched from Reddit")
            return False

    except Exception as e:
        print(f"❌ Reddit Error: {e}")
        return False


def main():
    print("🚀 Testing Both APIs...")
    print("=" * 50)

    binance_success = test_binance()
    reddit_success = test_reddit()

    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"Binance API: {'✅ SUCCESS' if binance_success else '❌ FAILED'}")
    print(f"Reddit API:  {'✅ SUCCESS' if reddit_success else '❌ FAILED'}")

    if binance_success and reddit_success:
        print("\n🎉 All APIs working! Ready to build the application.")
    else:
        print("\n🔧 Some APIs need troubleshooting.")


if __name__ == "__main__":
    main()