#!/usr/bin/env python3
"""
Top-level wrapper for Physics-Informed Koopman Neural Operator & DANN Architecture.
Imports BatteryKoopmanDANN from src/koopman/koopman_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.koopman_model import (
    GradientReversalFunction,
    KoopmanEncoder,
    KoopmanOperatorLayer,
    DomainDiscriminator,
    BatteryKoopmanDANN,
)

__all__ = [
    "GradientReversalFunction",
    "KoopmanEncoder",
    "KoopmanOperatorLayer",
    "DomainDiscriminator",
    "BatteryKoopmanDANN",
]
