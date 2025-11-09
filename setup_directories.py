#!/usr/bin/env python3
"""
Setup script for Quantum Trader AI - Creates all necessary directories and files
"""
import os


def setup_project():
    print("🚀 Setting up Quantum Trader AI project...")

    # Create directories
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        '.streamlit',
        'src/dashboard',
        'src/data',
        'src/models',
        'src/utils',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create .streamlit/config.toml
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
    with open(config_path, 'w') as f:
        f.write(config_content)
    print("✅ Created .streamlit/config.toml")

    # Create empty __init__.py files
    init_files = [
        'src/__init__.py',
        'src/dashboard/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py'
    ]

    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('')
        print(f"✅ Created: {init_file}")

    print("\n🎉 Project setup completed!")
    print("📁 All necessary directories and files created")
    print("🚀 You can now run: python run_dashboard.py")


if __name__ == "__main__":
    setup_project()