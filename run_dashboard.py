#!/usr/bin/env python3
"""
Simple script to run the Streamlit dashboard
"""
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from src.dashboard.app import main

if __name__ == "__main__":
    # This will be called by Streamlit
    print("🚀 Starting Crypto Sentiment Dashboard...")
    print("📊 Open your browser to http://localhost:8501")