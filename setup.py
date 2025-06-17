#!/usr/bin/env python3
"""
Setup script for the Optimized Fake News Detector
This script helps install dependencies and download required models.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Optimized Fake News Detector...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Some dependencies may have failed to install. Please check manually.")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  spaCy model download failed. You can install it manually later.")
    
    # Create necessary directories
    print("\n🔄 Creating directories...")
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    print("✅ Directories created!")
    
    # Test installation
    print("\n🔄 Testing installation...")
    try:
        import streamlit
        import pandas
        import sklearn
        import transformers
        import nltk
        print("✅ All major dependencies imported successfully!")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: streamlit run src/fake_news_detector.py")
    print("2. Open your browser to the displayed URL")
    print("3. Start analyzing news articles!")
    
    return True

if __name__ == "__main__":
    main()
