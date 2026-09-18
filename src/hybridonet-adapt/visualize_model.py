#!/usr/bin/env python3
"""
HybridoNet-Adapt — presentation-grade visual investigation.

Targets the specific findings from investigate_model.py:

  A. RUL trajectories per cell      -> reproduces the paper's Figure 9
  B. Anti-correlated head errors    -> WHY combining beats both heads
  C. Theta sweep                    -> WHY theta settles at ~0.5
  D. Per-head attention             -> is MHA really doing mean pooling?
  E. NODE action on real latents    -> what e^W does to actual data
  F. Error anatomy                  -> where the model fails, and how badly

Usage:
  python visualize_model.py \
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm
from scipy.stats import pearsonr
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

OUT = "figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def predict_all(model, X, device, bs=256):
    """Returns combined, source-head and target-head predictions (scaled 0-1)."""
    yc, ys, yt = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
            c, s, t, _ = model(xb)
            yc.append(c.cpu().numpy()); ys.append(s.cpu().numpy()); yt.append(t.cpu().numpy())
    return (np.vstack(yc).ravel(), np.vstack(ys).ravel(), np.vstack(yt).ravel())


# ======================================================================
# A. RUL trajectories per cell  (the paper's Figure 9)
# ======================================================================
def viz_trajectories(model, X, Y_raw, cells, scaler_y, device, n_cells=6):
    """
    For individual held-out cells, plot predicted vs observed RUL across the
    cell's life. This is THE standard battery-paper figure and is far more
    persuasive than a scatter plot: it shows whether the model tracks the
    degradation trajectory or just gets the average right.
    """
    print("\n[A] RUL trajectories per cell ...")
    uniq = np.unique(cells)
    # pick cells spanning short / medium / long life
    life = np.array([Y_raw[cells == c].max() for c in uniq])
    pick = uniq[np.argsort(life)][np.linspace(0, len(uniq) - 1, n_cells, dtype=int)]

    yc_all, _, _ = predict_all(model, X, device)
    yc_all = scaler_y.inverse_transform(yc_all)

    rows = int(np.ceil(n_cells / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, cid in zip(axes, pick):
        m = cells == cid
        yt_true, yt_pred = Y_raw[m], yc_all[m]
        order = np.argsort(-yt_true)                 # early life -> end of life
        cyc = np.arange(len(order))                  # window index along life
        ax.plot(cyc, yt_true[order], "k-", lw=2, label="Observed RUL")
        ax.plot(cyc, yt_pred[order], "-", c="tab:red", lw=1.5, alpha=0.85, label="Predicted")
        rmse = np.sqrt(mean_squared_error(yt_true, yt_pred))
        ax.set_title(f"{cid}\nRMSE = {rmse:.0f} cyc", fontsize=9)
        ax.set_xlabel("window index (early $\\rightarrow$ end of life)")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8)

    for ax in axes[len(pick):]:
        ax.axis("off")
    plt.suptitle("Per-cell RUL trajectories on blind test cells", y=1.00, fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/A_trajectories.png", bbox_inches="tight"); plt.close()
    print(f"    -> {OUT}/A_trajectories.png")
    print("    READ: a prediction that tracks the black line means the model follows")
    print("    degradation. A flat red line means it predicts the cell's average and")
    print("    the good RMSE is coming from the label range, not from real tracking.")


# ======================================================================
# B. Anti-correlated head errors  (why combining wins)
# ======================================================================
def viz_head_errors(model, X, Y_raw, scaler_y, device):
    print("\n[B] Head error structure ...")
    yc, ys, yt = predict_all(model, X, device)
    yc, ys, yt = (scaler_y.inverse_transform(v) for v in (yc, ys, yt))
    y = Y_raw[:len(yc)]

    es, et, ec = ys - y, yt - y, yc - y
    r, _ = pearsonr(es, et)

    fig = plt.figure(figsize=(15, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)

    # 1) the money plot: error of one head vs the other
    ax = fig.add_subplot(gs[0])
    ax.scatter(es, et, s=5, alpha=0.25, c="tab:purple")
    lim = max(np.abs(np.r_[es, et])) * 1.05
    ax.plot([-lim, lim], [lim, -lim], "r--", lw=1.2, label="perfect cancellation")
    ax.axhline(0, c="k", lw=0.6); ax.axvline(0, c="k", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("source-head error (cycles)")
    ax.set_ylabel("target-head error (cycles)")
    ax.set_title(f"Head errors are anti-correlated\nPearson r = {r:+.3f}")
    ax.legend(fontsize=8)

    # 2) error distributions
    ax = fig.add_subplot(gs[1])
    bins = np.linspace(-lim, lim, 60)
    ax.hist(es, bins=bins, alpha=0.5, label=f"source (RMSE {np.sqrt((es**2).mean()):.0f})")
    ax.hist(et, bins=bins, alpha=0.5, label=f"target (RMSE {np.sqrt((et**2).mean()):.0f})")
    ax.hist(ec, bins=bins, alpha=0.75, color="tab:green",
            label=f"combined (RMSE {np.sqrt((ec**2).mean()):.0f})")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("prediction error (cycles)"); ax.set_ylabel("count")
    ax.set_title("Error distributions")
    ax.legend(fontsize=8)

    # 3) both heads vs truth
    ax = fig.add_subplot(gs[2])
    o = np.argsort(y)
    ax.plot(y[o], ys[o], ".", ms=2, alpha=0.3, label="source head")
    ax.plot(y[o], yt[o], ".", ms=2, alpha=0.3, label="target head")
    ax.plot(y[o], yc[o], ".", ms=2, alpha=0.5, c="tab:green", label="combined")
    ax.plot(y[o], y[o], "k--", lw=1.2, label="ideal")
    ax.set_xlabel("true RUL (cycles)"); ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title("Each head brackets the truth")
    ax.legend(fontsize=8, markerscale=4)

    plt.savefig(f"{OUT}/B_head_errors.png", bbox_inches="tight"); plt.close()
    print(f"    Pearson r(source error, target error) = {r:+.4f}")
    print(f"    -> {OUT}/B_head_errors.png")
    print("    READ: r strongly NEGATIVE means the heads bracket the truth from")
    print("    opposite sides, so averaging cancels the error. That is the whole")
    print("    reason the dual-head design works, and it is why theta sits at ~0.5.")
    return ys, yt, y


# ======================================================================
# C. Theta sweep — is 0.5 actually optimal?
# ======================================================================
def viz_theta_sweep(ys, yt, y, learned_ts):
    print("\n[C] Theta sweep ...")
    ths = np.linspace(0, 1, 201)
    rmses = np.array([np.sqrt(mean_squared_error(y, t * ys + (1 - t) * yt)) for t in ths])
    best = ths[np.argmin(rmses)]

    plt.figure(figsize=(7.5, 4.6))
    plt.plot(ths, rmses, lw=2)
    plt.axvline(best, c="tab:green", ls="--", lw=1.5,
                label=f"empirical optimum $\\theta_S$={best:.3f} (RMSE {rmses.min():.0f})")
    plt.axvline(learned_ts, c="tab:red", ls=":", lw=2,
                label=f"learned $\\theta_S$={learned_ts:.3f}")
    plt.scatter([0, 1], [rmses[0], rmses[-1]], c="k", zorder=5)
    plt.annotate("target head only", (0, rmses[0]), textcoords="offset points",
                 xytext=(8, 8), fontsize=8)
    plt.annotate("source head only", (1, rmses[-1]), textcoords="offset points",
                 xytext=(-70, 8), fontsize=8)
    plt.xlabel("$\\theta_S$  (weight on source head)")
    plt.ylabel("target test RMSE (cycles)")
    plt.title("Why $\\theta$ settles near 0.5")
    plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/C_theta_sweep.png"); plt.close()
    print(f"    learned theta_S = {learned_ts:.4f} | empirical optimum = {best:.4f}")
    print(f"    RMSE at optimum = {rmses.min():.2f} | at theta=0 = {rmses[0]:.2f} | at theta=1 = {rmses[-1]:.2f}")
    print(f"    -> {OUT}/C_theta_sweep.png")
    print("    READ: if the learned theta sits in the valley, gradient descent found")
    print("    the right trade-off — theta was never 'stuck', 0.5 IS the optimum.")


# ======================================================================
# D. Per-head attention — is MHA really just averaging?
# ======================================================================
def viz_attention_perhead(model, X, device, n=512):
    print("\n[D] Per-head attention ...")
    fe = model.feature_extractor
    x = torch.tensor(X[:n], dtype=torch.float32, device=device)
    if x.dim() == 4:
        b, s, c, f = x.shape
        x = x.view(b, s, c * f)
    with torch.no_grad():
        lstm_out, _ = fe.lstm(x)
        # average_attn_weights=False keeps the 4 heads separate
        _, A = fe.mha(lstm_out, lstm_out, lstm_out,
                      need_weights=True, average_attn_weights=False)
    A = A.cpu().numpy()                        # (batch, heads, 10, 10)
    H = A.shape[1]
    Am = A.mean(axis=0)                        # (heads, 10, 10)

    fig, axes = plt.subplots(2, H, figsize=(3.4 * H, 6.6))
    for h in range(H):
        im = axes[0, h].imshow(Am[h], cmap="viridis", vmin=0, vmax=max(0.2, Am.max()))
        axes[0, h].set_title(f"head {h+1}: full matrix", fontsize=9)
        axes[0, h].set_xlabel("attends to"); axes[0, h].set_ylabel("from")
        plt.colorbar(im, ax=axes[0, h], fraction=0.046)

        last = Am[h, -1, :]
        ent = -(last * np.log(last + 1e-12)).sum()
        axes[1, h].bar(np.arange(1, 11), last, color="tab:blue")
        axes[1, h].axhline(0.1, c="r", ls="--", lw=1)
        axes[1, h].set_ylim(0, max(0.2, last.max() * 1.25))
        axes[1, h].set_title(f"head {h+1}: last row\nentropy {ent:.3f} / {np.log(10):.3f}", fontsize=9)
        axes[1, h].set_xlabel("cycle")
        print(f"    head {h+1}: last-row entropy = {ent:.4f}  "
              f"(uniform = {np.log(10):.4f})  max weight = {last.max():.4f}")

    plt.suptitle("Per-head attention (NOT averaged across heads)", y=1.0, fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/D_attention_perhead.png", bbox_inches="tight"); plt.close()
    print(f"    -> {OUT}/D_attention_perhead.png")
    print("    READ: if EVERY head is flat, MHA genuinely reduces to mean pooling.")
    print("    If individual heads are peaked but in different places, the earlier")
    print("    'flat' result was an averaging artifact and MHA does select.")


# ======================================================================
# E. What NODE does to real latent vectors
# ======================================================================
def viz_node_action(model, X, device, n=800):
    print("\n[E] NODE action on real latents ...")
    fe = model.feature_extractor
    x = torch.tensor(X[:n], dtype=torch.float32, device=device)
    if x.dim() == 4:
        b, s, c, f = x.shape
        x = x.view(b, s, c * f)
    with torch.no_grad():
        lstm_out, _ = fe.lstm(x)
        attn_out, _ = fe.mha(lstm_out, lstm_out, lstm_out)
        h = fe.layer_norm(lstm_out + attn_out)
        h_in = h[:, -1, :]                       # NODE input
        h_out = fe.node(h_in)                    # NODE output (pre-LayerNorm)
    hi, ho = h_in.cpu().numpy(), h_out.cpu().numpy()

    W = fe.node.ode_func.linear.weight.detach().cpu().numpy()
    eig = np.linalg.eigvals(expm(W))
    mag = np.abs(eig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    th = np.linspace(0, 2 * np.pi, 300)
    ax[0].plot(np.cos(th), np.sin(th), "--", c="gray", lw=1)
    sc = ax[0].scatter(eig.real, eig.imag, c=mag, cmap="coolwarm", s=22,
                       vmin=0.5, vmax=1.5, edgecolors="k", linewidths=0.2)
    ax[0].set_aspect("equal"); ax[0].axhline(0, c="k", lw=0.4); ax[0].axvline(0, c="k", lw=0.4)
    ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im")
    ax[0].set_title(f"Spectrum of $e^W$\n{(mag>1).sum()} amplify / {(mag<1).sum()} contract")
    plt.colorbar(sc, ax=ax[0], label="|$\\lambda$|")

    ax[1].scatter(np.abs(hi).mean(0), np.abs(ho).mean(0), s=14, alpha=0.7)
    m = max(np.abs(hi).mean(0).max(), np.abs(ho).mean(0).max())
    ax[1].plot([0, m], [0, m], "r--", lw=1, label="no change")
    ax[1].set_xlabel("mean |value| BEFORE NODE")
    ax[1].set_ylabel("mean |value| AFTER NODE")
    ax[1].set_title("Per-dimension magnitude change")
    ax[1].legend(fontsize=8)

    delta = np.linalg.norm(ho - hi, axis=1) / (np.linalg.norm(hi, axis=1) + 1e-9)
    ax[2].hist(delta, bins=45, color="tab:orange")
    ax[2].axvline(delta.mean(), c="k", ls="--", label=f"mean {delta.mean():.3f}")
    ax[2].set_xlabel("relative change  $\\|z_{out}-z_{in}\\| / \\|z_{in}\\|$")
    ax[2].set_ylabel("count")
    ax[2].set_title("How much NODE moves each sample")
    ax[2].legend(fontsize=8)

    plt.tight_layout(); plt.savefig(f"{OUT}/E_node_action.png"); plt.close()
    print(f"    mean relative change = {delta.mean():.4f} (0 = identity)")
    print(f"    -> {OUT}/E_node_action.png")
    print("    READ: this is NODE measured on real data rather than in theory.")
    print("    A large relative change confirms the block is not a pass-through.")


# ======================================================================
# F. Error anatomy — where does it fail?
# ======================================================================
def viz_error_anatomy(model, X, Y_raw, cells, scaler_y, device):
    print("\n[F] Error anatomy ...")
    yc, _, _ = predict_all(model, X, device)
    yc = scaler_y.inverse_transform(yc)
    y = Y_raw[:len(yc)]
    err = yc - y

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # error vs true RUL
    ax[0].scatter(y, err, s=4, alpha=0.2)
    ax[0].axhline(0, c="r", lw=1.2)
    bins = np.linspace(y.min(), y.max(), 15)
    idx = np.digitize(y, bins)
    bm = [err[idx == i].mean() if (idx == i).sum() > 5 else np.nan for i in range(1, len(bins))]
    ax[0].plot(bins[:-1], bm, "o-", c="tab:orange", lw=2, label="binned mean (bias)")
    ax[0].set_xlabel("true RUL (cycles)"); ax[0].set_ylabel("error (pred - true)")
    ax[0].set_title("Bias across the RUL range"); ax[0].legend(fontsize=8)

    # RMSE per bin
    br = [np.sqrt((err[idx == i] ** 2).mean()) if (idx == i).sum() > 5 else np.nan
          for i in range(1, len(bins))]
    ax[1].bar(bins[:-1], br, width=(bins[1] - bins[0]) * 0.85, color="tab:red", alpha=0.75)
    ax[1].set_xlabel("true RUL (cycles)"); ax[1].set_ylabel("RMSE (cycles)")
    ax[1].set_title("Where the error concentrates")

    # per-cell RMSE ranking
    uniq = np.unique(cells[:len(yc)])
    cr = np.array([np.sqrt(((yc[cells[:len(yc)] == c] - y[cells[:len(yc)] == c]) ** 2).mean())
                   for c in uniq])
    o = np.argsort(cr)
    ax[2].bar(range(len(cr)), cr[o], color="tab:blue")
    ax[2].axhline(cr.mean(), c="k", ls="--", label=f"mean {cr.mean():.0f} cyc")
    ax[2].set_xlabel("blind-test cell (sorted)"); ax[2].set_ylabel("per-cell RMSE (cycles)")
    ax[2].set_title(f"Per-cell spread: {cr.min():.0f} - {cr.max():.0f} cyc")
    ax[2].legend(fontsize=8)

    plt.tight_layout(); plt.savefig(f"{OUT}/F_error_anatomy.png"); plt.close()
    print(f"    per-cell RMSE: min={cr.min():.1f} median={np.median(cr):.1f} max={cr.max():.1f}")
    print(f"    -> {OUT}/F_error_anatomy.png")
    print("    READ: a sloped orange line means systematic bias (over-predicting at")
    print("    low RUL / under at high). A few tall bars in panel 3 means a handful")
    print("    of cells dominate your headline RMSE.")


# ======================================================================
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
    Xs, Ys, sc = s["X"], s["Y"], s["cell_ids"]
    Xt, Yt, tc = t["X"], t["Y"], t["cell_ids"]

    if args.severson_only and "matr" in args.source.lower():
        Xs, Ys, sc = filter_severson_cells(Xs, Ys, sc)
        print(f"--severson-only: {len(np.unique(sc))} source cells")

    Xtr, Xval, Ytr, Yval, _, _ = split_by_cell_id(Xs, Ys, sc, test_ratio=0.10, random_state=42)
    Xad, Xts, Yad, Yts, _, ts_cells = split_by_cell_id(Xt, Yt, tc, test_ratio=0.40, random_state=42)
    Xtr_s, Xval_s, Xad_s, Xts_s, _ = fit_and_transform_features_18d(Xtr, Xval, Xad, Xts)
    scaler_y = RobustRULScaler(y_max=args.rul_ceiling).fit()

    model = HybridoNetAdapt(input_dim=18, hidden_dim=args.hidden_dim,
                            num_lstm_layers=2, num_heads=4, dropout=0.1,
                            norm_type=args.norm_type)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.to(device).eval()
    print(f"loaded {args.checkpoint} (epoch {ck.get('best_epoch','?')}, "
          f"val RMSE {ck.get('best_val_rmse', float('nan')):.2f})")
    print(f"blind test: {len(Xts_s)} windows / {len(np.unique(ts_cells))} cells")

    viz_trajectories(model, Xts_s, Yts, ts_cells, scaler_y, device)
    ys, yt_, y = viz_head_errors(model, Xts_s, Yts, scaler_y, device)
    viz_theta_sweep(ys, yt_, y, model.theta_s.item())
    viz_attention_perhead(model, Xts_s, device)
    viz_node_action(model, Xts_s, device)
    viz_error_anatomy(model, Xts_s, Yts, ts_cells, scaler_y, device)

    print(f"\nDONE — figures in ./{OUT}/")


if __name__ == "__main__":
    main()
