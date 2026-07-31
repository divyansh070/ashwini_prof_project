#!/usr/bin/env python3
"""
Top-level wrapper for Domain Adversarial Transfer Learning Script.
Executes src/koopman/train_da_colab.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.train_da_colab import main

if __name__ == "__main__":
    main()
