#!/usr/bin/env python3
"""
Top-level wrapper for PatchTST Multi-Source Dataset Download Script.
Executes src/patchtst/download_datasets.py
"""
import sys
import os

# Add src/ to sys.path so patchtst imports work cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patchtst.download_datasets import main

if __name__ == "__main__":
    main()
