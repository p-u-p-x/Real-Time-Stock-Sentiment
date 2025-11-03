import subprocess
import sys


def install_packages():
    packages = [
        "textblob",
        "python-binance",
        "nltk",
        "plotly",
        "streamlit",
        "tqdm"
    ]

    print("🚀 Installing missing packages...")

    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

    print("\n🎉 All packages installed! Now testing...")


if __name__ == "__main__":
    install_packages()  # Fixed the function name