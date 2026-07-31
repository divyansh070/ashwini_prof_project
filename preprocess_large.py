#!/usr/bin/env python3
"""
Top-level wrapper for Large-Scale Universal SOC Preprocessing Script.
Executes src/koopman/preprocess_large.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.preprocess_large import main

if __name__ == "__main__":
    main()
