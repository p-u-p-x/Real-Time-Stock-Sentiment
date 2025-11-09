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
            'AAPL': ['apple', 'aapl', 'iphone', 'ipad', 'macbook', 'tim cook'],
            'TSLA': ['tesla', 'tsla', 'elon musk', 'model s', 'model 3', 'model x', 'model y', 'cybertruck'],
            'MSFT': ['microsoft', 'msft', 'windows', 'azure', 'surface', 'satya nadella', 'xbox'],
            'AMZN': ['amazon', 'amzn', 'jeff bezos', 'aws', 'prime', 'alexa'],
            'GOOGL': ['google', 'alphabet', 'googl', 'sundar pichai', 'android', 'youtube', 'search engine'],
            'NVDA': ['nvidia', 'nvda', 'jensen huang', 'gpu', 'rtx', 'ai chips', 'graphics card'],
            'META': ['meta', 'facebook', 'fb', 'mark zuckerberg', 'instagram', 'whatsapp', 'oculus'],
            'JPM': ['jpmorgan', 'jpm', 'chase', 'jamie dimon', 'bank', 'investment bank'],
            'NFLX': ['netflix', 'nflx', 'reed hastings', 'streaming', 'movies', 'tv shows'],
            'AMD': ['amd', 'advanced micro devices', 'lisa su', 'ryzen', 'processors', 'cpu']
        }

    def extract_symbols_from_text(self, text):
        """Extract cryptocurrency and stock symbols from news text"""
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

    def get_news_articles(self, query='stocks', language='en', max_articles=50):
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
        """Fetch from NewsAPI with enhanced stock coverage"""
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

            logger.info(f"Fetched {len(processed_articles)} news articles for query: {query}")
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

            logger.info(f"Fetched {len(processed_articles)} news articles from GNews for query: {query}")
            return processed_articles

        else:
            logger.error(f"GNews error: {response.status_code} - {response.text}")
            return []

    def get_comprehensive_news_sentiment(self):
        """Get comprehensive news sentiment for both crypto and stocks"""
        try:
            all_articles = []

            # Enhanced queries for both crypto and stocks
            crypto_queries = ['cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft']
            stock_queries = ['stocks', 'stock market', 'investing', 'earnings', 'trading', 'wall street']

            # Individual stock queries for better coverage
            individual_stocks = ['apple', 'tesla', 'microsoft', 'amazon', 'google', 'nvidia', 'meta', 'netflix', 'amd']

            all_queries = crypto_queries + stock_queries + individual_stocks

            for query in all_queries:
                articles = self.get_news_articles(query=query, max_articles=15)
                all_articles.extend(articles)
                time.sleep(0.5)  # Rate limiting

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

            logger.info(
                f"Processed sentiment for {len(sentiment_summary)} symbols from news (including {len([s for s in sentiment_summary if s['symbol'] in self.stock_keywords])} stocks)")
            return sentiment_summary, unique_articles

        except Exception as e:
            logger.error(f"Error processing comprehensive news sentiment: {e}")
            return [], []