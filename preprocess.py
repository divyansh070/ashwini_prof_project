#!/usr/bin/env python3
"""
Top-level wrapper for PatchTST Multi-Chemistry Preprocessing & SOD Grid Patching Script.
Executes src/patchtst/preprocess.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patchtst.preprocess import main

if __name__ == "__main__":
    main()
