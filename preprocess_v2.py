#!/usr/bin/env python3
"""
Top-level wrapper for Universal Domain Normalization (preprocess_v2.py).
Executes src/koopman/preprocess_v2.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.preprocess_v2 import main

if __name__ == "__main__":
    main()
