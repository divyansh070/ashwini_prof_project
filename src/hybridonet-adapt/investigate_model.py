#!/usr/bin/env python3
"""
HybridoNet-Adapt Layer-by-Layer Investigation.

Probes each architectural component of a TRAINED model and writes figures
suitable for a presentation:

  1. NODE operator      -> eigenvalue spectrum of e^W  (Koopman view)
  2. Attention          -> which of the 10 cycles the model actually attends to
  3. Latent space z     -> PCA of source vs target embeddings (paper Fig. 10)
  4. Predictor heads    -> source-head vs target-head disagreement, theta
  5. Input features     -> permutation importance over the 18 features

Usage:
  python investigate_model.py \
      --checkpoint checkpoints/hybrido_best.pt \
      --source data/hybridonet/processed/MATR_raw_features.npz \
      --target data/hybridonet/processed/HUST_raw_features.npz \
      --rul-ceiling 2500 --severson-only
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

from model_hybrido import HybridoNetAdapt
from train_hybrido import (
    split_by_cell_id,
    fit_and_transform_features_18d,
    RobustRULScaler,
    filter_severson_cells,
)


FEATURE_NAMES = [
    "V_mean", "V_std", "V_min", "V_max", "V_var", "V_med",
    "I_mean", "I_std", "I_min", "I_max", "I_var", "I_med",
    "Q_mean", "Q_std", "Q_min", "Q_max", "Q_var", "Q_med",
]

OUT = "figures"


# ----------------------------------------------------------------------
# 1. NODE: eigenvalue spectrum of the learned operator
# ----------------------------------------------------------------------
def probe_node(model):
    """
    The NODE block solves dz/dt = Wz from t=0 to t=1, which for a linear f
    is exactly multiplication by the matrix exponential e^W. Its eigenvalues
    tell you what the block actually DOES to the latent vector:
      |lambda| > 1  -> that direction is amplified
      |lambda| < 1  -> that direction is contracted (information discarded)
      |lambda| ~ 1  -> passed through (near-identity)
    """
    print("\n" + "=" * 70)
    print("1. NODE OPERATOR — eigenvalue spectrum of e^W")
    print("=" * 70)

    W = model.feature_extractor.node.ode_func.linear.weight.detach().cpu().numpy()
    b = model.feature_extractor.node.ode_func.linear.bias.detach().cpu().numpy()

    eigW = np.linalg.eigvals(W)
    E = expm(W)                      # the effective transform over t=0->1
    eigE = np.linalg.eigvals(E)
    mag = np.abs(eigE)

    print(f"  W shape: {W.shape},  ||W||_F = {np.linalg.norm(W):.4f},  ||b|| = {np.linalg.norm(b):.4f}")
    print(f"  e^W eigenvalue |lambda|:  min={mag.min():.4f}  median={np.median(mag):.4f}  max={mag.max():.4f}")
    print(f"  directions amplified (|lambda|>1): {int((mag > 1).sum())} / {len(mag)}")
    print(f"  directions contracted (|lambda|<1): {int((mag < 1).sum())} / {len(mag)}")
    print(f"  distance from identity: ||e^W - I||_F = {np.linalg.norm(E - np.eye(len(E))):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(np.cos(th), np.sin(th), "--", c="gray", lw=1, label="unit circle")
    ax[0].scatter(eigE.real, eigE.imag, s=18, alpha=0.7)
    ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im")
    ax[0].set_title("Eigenvalues of $e^W$ (NODE transform)")
    ax[0].axhline(0, c="k", lw=0.4); ax[0].axvline(0, c="k", lw=0.4)
    ax[0].legend(); ax[0].set_aspect("equal")

    ax[1].hist(mag, bins=40)
    ax[1].axvline(1.0, c="r", ls="--", label="identity (|$\\lambda$|=1)")
    ax[1].set_xlabel("|$\\lambda$|"); ax[1].set_ylabel("count")
    ax[1].set_title("Magnitude distribution")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/1_node_eigenvalues.png", dpi=150); plt.close()
    print(f"  -> {OUT}/1_node_eigenvalues.png")

    print("\n  INTERPRETATION: if most |lambda| cluster near 1, NODE is close to an")
    print("  identity map and contributes little — consistent with it being an")
    print("  empirically-motivated refinement rather than a load-bearing block.")
    print("  A wide spread means it is genuinely reshaping the latent space.")
    return mag


# ----------------------------------------------------------------------
# 2. Attention: which timesteps does the model use?
# ----------------------------------------------------------------------
def probe_attention(model, X, device, tag, n=512):
    """
    Captures the (10 x 10) attention weight matrix. Row -1 is what feeds
    'select last' -> the only row that actually reaches the predictors.
    """
    print("\n" + "=" * 70)
    print(f"2. ATTENTION — timestep usage ({tag})")
    print("=" * 70)

    fe = model.feature_extractor
    x = torch.tensor(X[:n], dtype=torch.float32, device=device)
    if x.dim() == 4:
        b, s, c, f = x.shape
        x = x.view(b, s, c * f)

    with torch.no_grad():
        lstm_out, _ = fe.lstm(x)
        _, attn_w = fe.mha(lstm_out, lstm_out, lstm_out,
                           need_weights=True, average_attn_weights=True)
    A = attn_w.cpu().numpy()             # (batch, 10, 10)
    last_row = A[:, -1, :].mean(axis=0)  # what the selected timestep attends to

    print("  Attention from the LAST timestep (the one selected) to each cycle:")
    for i, w in enumerate(last_row):
        bar = "#" * int(round(w * 60))
        print(f"    cycle {i+1:2d}: {w:.4f} {bar}")
    print(f"  entropy = {-(last_row * np.log(last_row + 1e-12)).sum():.3f} "
          f"(uniform would be {np.log(10):.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(A.mean(axis=0), cmap="viridis")
    ax[0].set_xlabel("attends to cycle"); ax[0].set_ylabel("from cycle")
    ax[0].set_title(f"Mean attention matrix ({tag})")
    plt.colorbar(im, ax=ax[0])

    ax[1].bar(np.arange(1, 11), last_row)
    ax[1].axhline(0.1, c="r", ls="--", label="uniform (0.1)")
    ax[1].set_xlabel("cycle in window"); ax[1].set_ylabel("attention weight")
    ax[1].set_title("Last timestep's attention")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/2_attention_{tag}.png", dpi=150); plt.close()
    print(f"  -> {OUT}/2_attention_{tag}.png")

    print("\n  INTERPRETATION: a flat bar chart means attention is averaging the")
    print("  window (MHA adds little beyond pooling). A peak on the recent cycles")
    print("  means it genuinely prioritises current state; a peak on cycle 1 means")
    print("  it is using the window's start as a degradation reference point.")
    return last_row


# ----------------------------------------------------------------------
# 3. Latent space: PCA of source vs target  (reproduces paper Fig. 10)
# ----------------------------------------------------------------------
def probe_latent(model, Xs, Xt, Ys, Yt, device, n=1500):
    print("\n" + "=" * 70)
    print("3. LATENT SPACE z — source vs target alignment (paper Fig. 10)")
    print("=" * 70)

    def emb(X):
        out = []
        with torch.no_grad():
            for i in range(0, min(len(X), n), 256):
                xb = torch.tensor(X[i:i + 256], dtype=torch.float32, device=device)
                out.append(model.extract_features(xb).cpu().numpy())
        return np.vstack(out)

    zs, zt = emb(Xs), emb(Xt)
    ys, yt = Ys[:len(zs)], Yt[:len(zt)]

    p = PCA(n_components=2).fit(np.vstack([zs, zt]))
    ps, pt = p.transform(zs), p.transform(zt)

    # how separable are the domains? (lower = better aligned)
    d_between = np.linalg.norm(zs.mean(0) - zt.mean(0))
    d_within = 0.5 * (zs.std(0).mean() + zt.std(0).mean())
    print(f"  latent dim: {zs.shape[1]},  PCA var explained: {p.explained_variance_ratio_.sum():.3f}")
    print(f"  ||mean(z_s) - mean(z_t)|| = {d_between:.4f}")
    print(f"  mean within-domain std     = {d_within:.4f}")
    print(f"  separation ratio           = {d_between / (d_within + 1e-9):.3f}  (lower = better aligned)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(ps[:, 0], ps[:, 1], s=6, alpha=0.4, label="Source", c="tab:blue")
    ax[0].scatter(pt[:, 0], pt[:, 1], s=6, alpha=0.4, label="Target", c="tab:green")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].set_title("Domain alignment (MMD effect)")
    ax[0].legend()

    sc = ax[1].scatter(np.vstack([ps, pt])[:, 0], np.vstack([ps, pt])[:, 1],
                       c=np.concatenate([ys, yt]), s=6, alpha=0.6, cmap="plasma")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].set_title("Same embedding, coloured by true RUL")
    plt.colorbar(sc, ax=ax[1], label="RUL (cycles)")
    plt.tight_layout(); plt.savefig(f"{OUT}/3_latent_pca.png", dpi=150); plt.close()
    print(f"  -> {OUT}/3_latent_pca.png")

    print("\n  INTERPRETATION: left panel = did MMD work (overlapping clouds = yes).")
    print("  RIGHT PANEL IS THE IMPORTANT ONE: if colour varies smoothly along some")
    print("  direction, the latent space encodes degradation continuously — that is")
    print("  the actual evidence the representation is meaningful, not just aligned.")
    return zs, zt


# ----------------------------------------------------------------------
# 4. Predictor heads + theta
# ----------------------------------------------------------------------
def probe_heads(model, Xt, Yt_raw, scaler_y, device, n=2000):
    print("\n" + "=" * 70)
    print("4. PREDICTOR HEADS — source vs target head, and theta")
    print("=" * 70)

    ys_l, yt_l, yc_l = [], [], []
    with torch.no_grad():
        for i in range(0, min(len(Xt), n), 256):
            xb = torch.tensor(Xt[i:i + 256], dtype=torch.float32, device=device)
            yc, yhs, yht, _ = model(xb)
            yc_l.append(yc.cpu().numpy()); ys_l.append(yhs.cpu().numpy()); yt_l.append(yht.cpu().numpy())

    yc = scaler_y.inverse_transform(np.vstack(yc_l).ravel())
    yhs = scaler_y.inverse_transform(np.vstack(ys_l).ravel())
    yht = scaler_y.inverse_transform(np.vstack(yt_l).ravel())
    y = Yt_raw[:len(yc)]

    ts, tt = model.theta_s.item(), model.theta_t.item()
    print(f"  theta_S = {ts:.4f}   theta_T = {tt:.4f}   (sum = {ts + tt:.4f})")
    for name, pred in [("source head alone", yhs), ("target head alone", yht), ("COMBINED", yc)]:
        rmse = np.sqrt(mean_squared_error(y, pred))
        print(f"    {name:20s} RMSE = {rmse:8.2f} cyc")
    print(f"  head disagreement |y_s - y_t|: mean={np.abs(yhs - yht).mean():.1f} "
          f"max={np.abs(yhs - yht).max():.1f} cycles")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y, yc, s=5, alpha=0.3)
    lim = [0, max(y.max(), yc.max())]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_xlabel("true RUL"); ax[0].set_ylabel("predicted RUL")
    ax[0].set_title("Combined prediction vs truth")

    ax[1].scatter(yhs, yht, s=5, alpha=0.3)
    lim2 = [min(yhs.min(), yht.min()), max(yhs.max(), yht.max())]
    ax[1].plot(lim2, lim2, "r--", lw=1)
    ax[1].set_xlabel("source head $\\hat{y}_s$"); ax[1].set_ylabel("target head $\\hat{y}_t$")
    ax[1].set_title(f"Head agreement ($\\theta_S$={ts:.3f}, $\\theta_T$={tt:.3f})")
    plt.tight_layout(); plt.savefig(f"{OUT}/4_heads.png", dpi=150); plt.close()
    print(f"  -> {OUT}/4_heads.png")

    print("\n  INTERPRETATION: if the two heads agree closely (points on the diagonal),")
    print("  theta barely matters and the dual-head design is not earning its place.")
    print("  If COMBINED beats both heads alone, the trade-off mechanism is working.")


# ----------------------------------------------------------------------
# 5. Input features: permutation importance
# ----------------------------------------------------------------------
def probe_features(model, Xt, Yt_raw, scaler_y, device, n=2000, seed=0):
    """
    Shuffle one of the 18 features across samples and measure RMSE degradation.
    Large increase = the model depends on that feature.
    """
    print("\n" + "=" * 70)
    print("5. INPUT FEATURES — permutation importance")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    X = Xt[:n].copy()
    y = Yt_raw[:n]

    def rmse_of(Xa):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xa), 256):
                xb = torch.tensor(Xa[i:i + 256], dtype=torch.float32, device=device)
                yc, _, _, _ = model(xb)
                out.append(yc.cpu().numpy())
        pred = scaler_y.inverse_transform(np.vstack(out).ravel())
        return float(np.sqrt(mean_squared_error(y[:len(pred)], pred)))

    base = rmse_of(X)
    print(f"  baseline RMSE = {base:.2f} cyc\n")

    imps = []
    for f in range(18):
        Xp = X.copy()
        c, k = divmod(f, 6)                      # (N,10,3,6) layout
        Xp[:, :, c, k] = Xp[rng.permutation(len(Xp)), :, c, k]
        imps.append(rmse_of(Xp) - base)

    order = np.argsort(imps)[::-1]
    for i in order:
        bar = "#" * int(max(0, imps[i]) / max(1e-9, max(imps)) * 50)
        print(f"    {FEATURE_NAMES[i]:8s} +{imps[i]:8.2f} cyc  {bar}")

    plt.figure(figsize=(9, 5))
    plt.barh([FEATURE_NAMES[i] for i in order[::-1]], [imps[i] for i in order[::-1]])
    plt.xlabel("RMSE increase when shuffled (cycles)")
    plt.title("Permutation importance of the 18 input features")
    plt.tight_layout(); plt.savefig(f"{OUT}/5_feature_importance.png", dpi=150); plt.close()
    print(f"\n  -> {OUT}/5_feature_importance.png")

    print("\n  INTERPRETATION: you expect the capacity features (esp. Q_max/Q_mean)")
    print("  to dominate, since your diagnostic measured r(RUL, Q_max) = 0.98.")
    print("  If they DON'T, the model is relying on something other than the")
    print("  physical degradation signal — worth knowing before you present it.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/hybrido_best.pt")
    ap.add_argument("--source", default="data/hybridonet/processed/MATR_raw_features.npz")
    ap.add_argument("--target", default="data/hybridonet/processed/HUST_raw_features.npz")
    ap.add_argument("--rul-ceiling", type=float, default=2500.0)
    ap.add_argument("--severson-only", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--norm-type", default="layernorm")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    s = np.load(args.source, allow_pickle=True)
    t = np.load(args.target, allow_pickle=True)
    Xs_all, Ys_all, sc_ids = s["X"], s["Y"], s["cell_ids"]
    Xt_all, Yt_all, tc_ids = t["X"], t["Y"], t["cell_ids"]

    if args.severson_only and "matr" in args.source.lower():
        Xs_all, Ys_all, sc_ids = filter_severson_cells(Xs_all, Ys_all, sc_ids)
        print(f"--severson-only: Filtered MATR to {len(np.unique(sc_ids))} Severson et al. 2019 cells (batches 1-3).")


    # same splits as training (random_state=42)
    Xtr, Xval, Ytr, Yval, _, _ = split_by_cell_id(Xs_all, Ys_all, sc_ids, test_ratio=0.10, random_state=42)
    Xad, Xts, Yad, Yts, _, _ = split_by_cell_id(Xt_all, Yt_all, tc_ids, test_ratio=0.40, random_state=42)

    Xtr_s, Xval_s, Xad_s, Xts_s, _ = fit_and_transform_features_18d(Xtr, Xval, Xad, Xts)
    scaler_y = RobustRULScaler(y_max=args.rul_ceiling).fit()

    model = HybridoNetAdapt(input_dim=18, hidden_dim=args.hidden_dim,
                            num_lstm_layers=2, num_heads=4, dropout=0.1,
                            norm_type=args.norm_type)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.to(device).eval()
    print(f"loaded {args.checkpoint} (best epoch {ck.get('best_epoch', '?')}, "
          f"val RMSE {ck.get('best_val_rmse', float('nan')):.2f})")

    probe_node(model)
    probe_attention(model, Xtr_s, device, "source")
    probe_attention(model, Xts_s, device, "target")
    probe_latent(model, Xtr_s, Xts_s, Ytr, Yts, device)
    probe_heads(model, Xts_s, Yts, scaler_y, device)
    probe_features(model, Xts_s, Yts, scaler_y, device)

    print("\n" + "=" * 70)
    print(f"DONE — figures in ./{OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()