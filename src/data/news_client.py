import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentCollector:
    def __init__(self, api_key, provider='newsapi'):
        self.api_key = api_key
        self.provider = provider
        self.base_url = "https://newsapi.org/v2" if provider == 'newsapi' else "https://gnews.io/api/v4"

    def extract_symbols_from_text(self, text):
        """Extract cryptocurrency symbols from news text"""
        if not text:
            return []

        text_lower = text.lower()
        mentioned_symbols = []

        crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi'],
            'ETH': ['ethereum', 'eth', 'vitalik'],
            'ADA': ['cardano', 'ada', 'charles hoskinson'],
            'DOT': ['polkadot', 'dot', 'gavin wood'],
            'LINK': ['chainlink', 'link', 'oracle'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'DOGE': ['dogecoin', 'doge']
        }

        for symbol, keywords in crypto_keywords.items():
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

    def get_news_articles(self, query='cryptocurrency', language='en', max_articles=50):
        """Fetch news articles from API"""
        try:
            if self.provider == 'newsapi':
                return self._get_newsapi_articles(query, language, max_articles)
            else:
                return self._get_gnews_articles(query, language, max_articles)

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _get_newsapi_articles(self, query, language, max_articles):
        """Fetch from NewsAPI"""
        url = f"{self.base_url}/everything"

        params = {
            'q': query,
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key,
            'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            processed_articles = []
            for article in articles:
                # Analyze sentiment and extract symbols
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"

                sentiment = self.analyze_sentiment(content)
                symbols = self.extract_symbols_from_text(content)

                processed_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment': sentiment,
                    'symbols_mentioned': symbols,
                    'content': content[:500]  # First 500 chars
                }
                processed_articles.append(processed_article)

            logger.info(f"Fetched {len(processed_articles)} news articles")
            return processed_articles

        else:
            logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
            return []

    def _get_gnews_articles(self, query, language, max_articles):
        """Fetch from GNews API"""
        url = f"{self.base_url}/search"

        params = {
            'q': query,
            'lang': language,
            'max': min(max_articles, 100),
            'apikey': self.api_key,
            'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            processed_articles = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"

                sentiment = self.analyze_sentiment(content)
                symbols = self.extract_symbols_from_text(content)

                processed_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment': sentiment,
                    'symbols_mentioned': symbols,
                    'content': content[:500]
                }
                processed_articles.append(processed_article)

            logger.info(f"Fetched {len(processed_articles)} news articles from GNews")
            return processed_articles

        else:
            logger.error(f"GNews error: {response.status_code} - {response.text}")
            return []

    def get_crypto_news_sentiment(self):
        """Get comprehensive crypto news sentiment"""
        try:
            all_articles = []

            # Fetch articles for different crypto-related queries
            queries = ['cryptocurrency', 'bitcoin', 'ethereum', 'blockchain']

            for query in queries:
                articles = self.get_news_articles(query=query, max_articles=20)
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting

            # Remove duplicates based on title
            unique_articles = {}
            for article in all_articles:
                title = article['title']
                if title not in unique_articles:
                    unique_articles[title] = article

            unique_articles = list(unique_articles.values())

            # Calculate sentiment by symbol
            symbol_sentiment = {}

            for article in unique_articles:
                for symbol in article['symbols_mentioned']:
                    if symbol not in symbol_sentiment:
                        symbol_sentiment[symbol] = {
                            'total_sentiment': 0,
                            'article_count': 0,
                            'mentions': 0
                        }

                    symbol_sentiment[symbol]['total_sentiment'] += article['sentiment']
                    symbol_sentiment[symbol]['article_count'] += 1
                    symbol_sentiment[symbol]['mentions'] += 1

            # Create sentiment summary
            sentiment_summary = []
            current_time = datetime.now()

            for symbol, data in symbol_sentiment.items():
                if data['article_count'] > 0:
                    avg_sentiment = data['total_sentiment'] / data['article_count']

                    sentiment_summary.append({
                        'symbol': symbol,
                        'avg_sentiment': avg_sentiment,
                        'total_mentions': data['mentions'],
                        'article_count': data['article_count'],
                        'timestamp': current_time,
                        'source': 'news'
                    })

            logger.info(f"Processed sentiment for {len(sentiment_summary)} symbols from news")
            return sentiment_summary, unique_articles

        except Exception as e:
            logger.error(f"Error processing crypto news sentiment: {e}")
            return [], []