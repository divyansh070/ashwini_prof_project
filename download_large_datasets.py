#!/usr/bin/env python3
"""
Top-level wrapper for Large-Scale Battery Dataset Acquisition Script.
Executes src/download_large_datasets.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from download_large_datasets import main

if __name__ == "__main__":
    main()
