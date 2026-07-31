#!/usr/bin/env python3
"""
Top-level wrapper for Large-Scale Koopman Neural Operator Benchmark Script.
Executes src/koopman/train_large_benchmarks.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.train_large_benchmarks import main

if __name__ == "__main__":
    main()
