#!/usr/bin/env python3
"""
Cross-Chemistry Visualizer (Stage 0).

Reuses the presentation-grade plotting routines from `visualize_model.py`
without code duplication, adding support for dynamic input dimensions (18-D, 30-D, 36-D).

Generates the key diagnostic figures:
  A. RUL trajectories per cell      -> checks if initial predictions spread with lifespan
  B. Anti-correlated head errors    -> verifies source/target head error synergy
  C. Theta sweep                    -> examines convex combination optimality
  D. Per-head attention             -> analyzes temporal focus
  E. NODE action on latents         -> evaluates continuous dynamics
  F. Error anatomy                  -> checks if bias curve flattens across lifespan
"""

import os
import sys
import argparse
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
hybridonet_dir = os.path.join(os.path.dirname(current_dir), "hybridonet-adapt")

for p in [current_dir, hybridonet_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_model
from model_xchem import HybridoNetAdapt
from train_xchem import (
    split_by_cell_id,
    fit_and_transform_features,
    RobustRULScaler,
    filter_severson_cells,
    resolve_file_path,
)


def main():
    ap = argparse.ArgumentParser(description="Cross-Chemistry Presentation Visualizer")
    ap.add_argument("--checkpoint", default="checkpoints/xchem_best.pt", help="Path to trained checkpoint")
    ap.add_argument("--source", default="data/hybridonet/processed/MATR_raw_features.npz", help="Path to source .npz")
    ap.add_argument("--target", default="data/hybridonet/processed/HUST_raw_features.npz", help="Path to target .npz")
    ap.add_argument("--rul-ceiling", type=float, default=2500.0, help="Physical RUL ceiling (default: 2500)")
    ap.add_argument("--severson-only", action="store_true", help="Filter MATR to 124 Severson et al. 2019 cells")
    ap.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    ap.add_argument("--norm-type", default="layernorm", help="Predictor norm type (default: layernorm)")
    ap.add_argument("--out-dir", default="figures", help="Directory to save figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    visualize_model.OUT = args.out_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    src_file = resolve_file_path(args.source)
    tgt_file = resolve_file_path(args.target)

    s = np.load(src_file, allow_pickle=True)
    t = np.load(tgt_file, allow_pickle=True)
    Xs, Ys, sc = s["X"], s["Y"], s["cell_ids"]
    Xt, Yt, tc = t["X"], t["Y"], t["cell_ids"]

    # Dynamic feature dimension detection
    if "num_features" in s:
        input_dim = int(s["num_features"])
    else:
        input_dim = int(np.prod(Xs.shape[2:]))
    print(f"Detected input feature dimension: {input_dim}-D")

    if args.severson_only and "matr" in src_file.lower():
        Xs, Ys, sc = filter_severson_cells(Xs, Ys, sc)
        print(f"--severson-only: {len(np.unique(sc))} source cells")

    Xtr, Xval, Ytr, Yval, _, _ = split_by_cell_id(Xs, Ys, sc, test_ratio=0.10, random_state=42)
    Xad, Xts, Yad, Yts, _, ts_cells = split_by_cell_id(Xt, Yt, tc, test_ratio=0.40, random_state=42)

    Xtr_s, Xval_s, Xad_s, Xts_s, _ = fit_and_transform_features(Xtr, Xval, Xad, Xts)
    scaler_y = RobustRULScaler(y_max=args.rul_ceiling).fit(Ytr, Yval, Yad)

    ckpt_path = resolve_file_path(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ck = torch.load(ckpt_path, map_location=device)
    state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

    # Detect input_dim from checkpoint if available
    if isinstance(ck, dict) and "input_dim" in ck:
        ckpt_input_dim = ck["input_dim"]
        if ckpt_input_dim != input_dim:
            print(f"Warning: Checkpoint input_dim ({ckpt_input_dim}) != data input_dim ({input_dim}). Using checkpoint dim.")
            input_dim = ckpt_input_dim

    model = HybridoNetAdapt(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.1,
        norm_type=args.norm_type
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    best_epoch = ck.get("best_epoch", "?") if isinstance(ck, dict) else "?"
    best_val_rmse = ck.get("best_val_rmse", float("nan")) if isinstance(ck, dict) else float("nan")
    print(f"Loaded {ckpt_path} (epoch {best_epoch}, val RMSE {best_val_rmse:.2f} cyc)")
    print(f"Target blind test set: {len(Xts_s)} windows / {len(np.unique(ts_cells))} cells")

    # Call visualization functions imported directly from visualize_model
    visualize_model.viz_trajectories(model, Xts_s, Yts, ts_cells, scaler_y, device)
    ys, yt_, y = visualize_model.viz_head_errors(model, Xts_s, Yts, scaler_y, device)
    visualize_model.viz_theta_sweep(ys, yt_, y, model.theta_s.item())
    visualize_model.viz_attention_perhead(model, Xts_s, device)
    visualize_model.viz_node_action(model, Xts_s, device)
    visualize_model.viz_error_anatomy(model, Xts_s, Yts, ts_cells, scaler_y, device)

    print(f"\nDONE — all figures successfully generated in {args.out_dir}/")


if __name__ == "__main__":
    main()
