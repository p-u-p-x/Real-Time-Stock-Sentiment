import os
from dotenv import load_dotenv

load_dotenv()


def test_basic_imports():
    print("🧪 Testing Basic Imports...")

    try:
        import pandas as pd
        print("✅ pandas - OK")
    except ImportError as e:
        print(f"❌ pandas - {e}")

    try:
        import praw
        print("✅ praw - OK")
    except ImportError as e:
        print(f"❌ praw - {e}")

    try:
        from binance.client import Client
        print("✅ python-binance - OK")
    except ImportError as e:
        print(f"❌ python-binance - {e}")

    try:
        from textblob import TextBlob
        print("✅ textblob - OK")
    except ImportError as e:
        print(f"❌ textblob - {e}")


def test_env_variables():
    print("\n🔑 Testing Environment Variables...")

    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']

    for var in required_vars:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var} - SET")
        else:
            print(f"❌ {var} - MISSING")


if __name__ == "__main__":
    print("🚀 Quick Installation Check")
    print("=" * 40)
    test_basic_imports()
    test_env_variables()
    print("\n🎯 Next steps will be based on this output!")