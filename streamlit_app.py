#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit App Entry Point
"""
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from src.fake_news_detector import main

if __name__ == "__main__":
    main()
