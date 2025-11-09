#!/usr/bin/env python3
"""
Simple script to run the Streamlit dashboard - DEPLOYMENT FIXED
"""
import os
import sys
import subprocess


def main():
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        '.streamlit'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Create config.toml if it doesn't exist
    config_content = """[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "localhost"
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[runner]
magicEnabled = false
"""

    config_path = os.path.join('.streamlit', 'config.toml')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Created .streamlit/config.toml")

    print("🚀 Starting Quantum Trader AI Dashboard...")
    print("📊 Open your browser to http://localhost:8501")
    print("⏳ Please wait while the dashboard loads...")

    # Run streamlit with the app
    dashboard_path = os.path.join('src', 'dashboard', 'app.py')

    if os.path.exists(dashboard_path):
        try:
            # Use subprocess to run streamlit
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                dashboard_path,
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running dashboard: {e}")
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
    else:
        print(f"❌ Dashboard file not found at: {dashboard_path}")
        print("💡 Make sure your project structure is correct")


if __name__ == "__main__":
    main()