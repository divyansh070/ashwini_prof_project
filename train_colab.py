#!/usr/bin/env python3
"""
Top-level wrapper for Google Colab GPU Multi-Dataset Transfer Learning Script.
Executes src/patchtst/train_colab.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patchtst.train_colab import main

if __name__ == "__main__":
    main()
