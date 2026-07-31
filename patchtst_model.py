#!/usr/bin/env python3
"""
Top-level wrapper exporting BatteryPatchTST from src/patchtst/patchtst_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from patchtst.patchtst_model import BatteryPatchTST, PatchEmbedding, RevIN

if __name__ == "__main__":
    import torch
    model = BatteryPatchTST(num_channels=46, seq_len=200, d_model=64, nhead=4, num_layers=4)
    dummy_input = torch.randn(8, 46, 200)
    out = model(dummy_input)
    print(f"PatchTST test forward successful! Output shape: {out.shape}")
    model.freeze_encoder()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen encoder trainable head params: {trainable_params:,}")
