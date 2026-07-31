#!/usr/bin/env python3
"""
Top-level wrapper exporting BatteryKoopmanDANN from src/koopman/koopman_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from koopman.koopman_model import BatteryKoopmanDANN, KoopmanEncoder, KoopmanOperatorLayer, SelfAttentionModule, DomainDiscriminator, grad_reverse

if __name__ == "__main__":
    import torch
    model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64)
    dummy_input = torch.randn(8, 46, 200)
    pred_log, dom_logits, kno_loss = model(dummy_input, alpha=0.5)
    print("Koopman DANN test forward successful!")
    print(f"  RUL Pred shape    : {pred_log.shape}")
    print(f"  Domain Logits     : {dom_logits.shape}")
    print(f"  KNO Linearity Loss: {kno_loss.item():.4f}")
