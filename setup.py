from setuptools import setup, find_packages

setup(
    name="quantum-trader-ai",
    version="1.0.0",
    description="Advanced Stock & Crypto Analytics Platform",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.3",
        "plotly>=5.15.0",
        "yfinance>=0.2.18",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "textblob>=0.17.1",
        "python-binance>=1.0.19",
        "praw>=7.7.1",
        "scikit-learn>=1.2.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)