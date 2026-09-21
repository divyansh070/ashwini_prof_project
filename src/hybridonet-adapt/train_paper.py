#!/usr/bin/env python3
"""
HybridoNet-Adapt — paper-faithful training & evaluation
(Tran et al., 2025, arXiv:2503.21392v2, Table 2 "All" / Table 3 setting).

Protocol reproduced from the paper
----------------------------------
  Source      : training data of the first dataset (TRI / MATR training set, 41 cells)
  Target      : all training cells of the second dataset (LHP / HUST, 55 cells)
  Blind test  : the 22 LHP test cells listed by channel in Table 3
  Split       : training data divided 90% train / 10% validation (Sec. 4.1)
  Selection   : checkpoint with lowest validation RMSE (Sec. 4.1)
  Optimizer   : AdamW, fixed LR 0.0005, batch 128, 10 epochs (Sec. 4.1)
  Repeats     : 10 runs; final prediction = average of the runs (Sec. 4.1)
  Loss        : MSE(Y_S, y_S) + MSE(Y_T, y_T) + lambda * MMD(G_F(X_S), G_F(X_T))
                with Y_S = source head alone (Eq. 2), Y_T = theta-blend (Eq. 1)
  lambda      : 2 / (1 + exp(-10 * epoch/epochs)) - 1 (Sec. 4.1)
  MMD         : single Gaussian kernel exp(-||x-y||^2 / (2 sigma^2)) (Eq. 3)
  Metrics     : RMSE, R^2, MAPE = mean(|y - yhat| / cycle_life) * 100 (Sec. 4.3)

!! How the paper aggregates metrics (verified against Table 3):
   The headline numbers (RMSE 153.24, R^2 0.88, MAPE 7.30%) are the MEAN OF
   PER-CELL metrics over the 22 test cells — the "Mean" row of Table 3 — not
   errors pooled over every window. This script reports both; compare the
   per-cell ("paper") numbers against the paper.

Details the paper leaves unspecified are flags, each marked [UNSPECIFIED].
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_paper import HybridoNetAdapt  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [Paper] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("Paper")


# ----------------------------------------------------------------------------
# Paper constants
# ----------------------------------------------------------------------------
# Table 3: the 22 LHP test cells (by protocol channel) and the paper's
# per-cell HybridoNet-Adapt results, used for a side-by-side comparison.
PAPER_TEST_CELLS: Dict[str, Tuple[float, float, float]] = {
    #  channel : (RMSE, R2, MAPE%)
    "1-1": (57.84, 0.98, 3.39),   "1-2": (514.39, 0.55, 17.34),
    "2-5": (158.42, 0.84, 9.38),  "3-1": (129.66, 0.94, 6.42),
    "4-5": (125.98, 0.92, 7.40),  "5-3": (392.70, 0.74, 12.40),
    "6-1": (34.57, 0.99, 1.69),   "6-2": (104.43, 0.96, 5.03),
    "6-6": (248.97, 0.87, 9.52),  "6-8": (217.09, 0.90, 8.30),
    "7-5": (142.18, 0.93, 6.52),  "7-6": (81.06, 0.96, 5.00),
    "8-1": (319.56, 0.25, 23.94), "8-5": (227.22, 0.64, 16.50),
    "8-6": (116.28, 0.97, 4.29),  "8-8": (49.44, 0.99, 2.31),
    "9-4": (34.23, 1.00, 1.43),   "9-6": (56.22, 0.99, 2.97),
    "10-1": (57.33, 0.99, 2.52),  "10-4": (43.13, 0.99, 2.05),
    "10-6": (103.52, 0.97, 3.96), "10-7": (157.16, 0.90, 8.28),
}
PAPER_HEADLINE = {"rmse": 153.24, "r2": 0.88, "mape": 7.30}


# ----------------------------------------------------------------------------
# Cell selection
# ----------------------------------------------------------------------------
def hust_channel(cell_id: str) -> Optional[str]:
    """'hust_HUST_10-4' -> '10-4'. Returns None if no channel pattern found."""
    m = re.search(r"(\d+-\d+)", str(cell_id))
    return m.group(1) if m else None


def resolve_file_path(path: str) -> str:
    """Resolve file path with case-insensitivity and TRI/MATR or LHP/HUST aliases."""
    if os.path.exists(path):
        return path
    dirname, basename = os.path.split(path)
    dirname = dirname or "."
    if os.path.exists(dirname):
        for f in os.listdir(dirname):
            if f.lower() == basename.lower():
                return os.path.join(dirname, f)
    aliases = []
    if "tri" in path.lower():
        aliases.append(re.sub(r"tri", "MATR", path, flags=re.IGNORECASE))
        aliases.append(re.sub(r"tri", "matr", path, flags=re.IGNORECASE))
    elif "matr" in path.lower():
        aliases.append(re.sub(r"matr", "TRI", path, flags=re.IGNORECASE))
        aliases.append(re.sub(r"matr", "tri", path, flags=re.IGNORECASE))
    if "lhp" in path.lower():
        aliases.append(re.sub(r"lhp", "HUST", path, flags=re.IGNORECASE))
        aliases.append(re.sub(r"lhp", "hust", path, flags=re.IGNORECASE))
    elif "hust" in path.lower():
        aliases.append(re.sub(r"hust", "LHP", path, flags=re.IGNORECASE))
        aliases.append(re.sub(r"hust", "lhp", path, flags=re.IGNORECASE))
    for alt in aliases:
        if os.path.exists(alt):
            return alt
    return path


def matr_batch_cell(cell_id: str) -> Optional[Tuple[int, int]]:
    """'..._b2c17' -> (2, 17)."""
    m = re.search(r"b(\d+)c(\d+)", str(cell_id).lower())
    return (int(m.group(1)), int(m.group(2))) if m else None


def select_source_cells(cell_ids: np.ndarray, mode: str,
                        cells_file: Optional[str]) -> np.ndarray:
    """
    Returns the unique source cell ids to train on.

      severson-train : Severson et al. 2019 training split (41 cells). Reproduces
                       the split rule from Severson's released code over batches
                       1+2 ordered by (batch, cell): train = odd indices.
                       BEST-EFFORT — depends on how the mirror orders and prunes
                       cells. A warning is printed if the count is not 41.
      severson-all   : all batch 1-3 cells (124), excluding Attia batch 4.
      file           : one cell id per line in --source-cells-file (exact ids).
    """
    uniq = np.unique(cell_ids)
    if mode == "file":
        if not cells_file:
            raise ValueError("--source-cells file requires --source-cells-file")
        wanted = {l.strip() for l in open(cells_file) if l.strip()}
        sel = np.array([c for c in uniq if c in wanted])
        missing = wanted - set(sel.tolist())
        if missing:
            log.warning(f"{len(missing)} ids in {cells_file} not found, e.g. {sorted(missing)[:3]}")
        return sel

    parsed = [(c, matr_batch_cell(c)) for c in uniq]
    parsed = [(c, bc) for c, bc in parsed if bc is not None]
    if not parsed:
        raise ValueError("Could not parse MATR batch/cell ids (expected '...b<batch>c<cell>'). "
                         "Use --source-cells file with explicit ids.")

    if mode == "severson-all":
        return np.array(sorted([c for c, (b, _) in parsed if b in (1, 2, 3)]))

    if mode == "severson-train":
        b12 = sorted([(bc, c) for c, bc in parsed if bc[0] in (1, 2)])
        ordered = [c for _, c in b12]
        n = len(ordered)
        train = [ordered[i] for i in range(1, n - 1, 2)]
        if len(train) != 41:
            log.warning(f"severson-train produced {len(train)} cells (paper: 41). Batches 1+2 "
                        f"have {n} cells in this mirror (Severson used 84). Verify, or pass "
                        f"--source-cells file with the exact 41 ids.")
        return np.array(train)

    raise ValueError(f"unknown --source-cells mode: {mode}")


def split_cells(cells: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-level split: returns (keep, held_out) with round(frac*n) held out (>=1)."""
    rng = np.random.RandomState(seed)
    cells = np.array(sorted(cells))
    k = max(1, int(round(frac * len(cells))))
    held = rng.choice(cells, size=k, replace=False)
    keep = np.array([c for c in cells if c not in set(held)])
    return keep, np.array(sorted(held))


# ----------------------------------------------------------------------------
# MMD — single Gaussian kernel, paper Eq. (3)
# ----------------------------------------------------------------------------
class GaussianMMD(nn.Module):
    """
    L = mean k(xs,xs') + mean k(xt,xt') - 2 mean k(xs,xt),
    k(x,y) = exp(-||x-y||^2 / (2 sigma^2)).

    sigma [UNSPECIFIED in paper]:
      sigma > 0  -> fixed bandwidth
      sigma <= 0 -> median heuristic: 2 sigma^2 = median pairwise squared distance
                    of the current batch (detached). Default.
    """

    def __init__(self, sigma: float = -1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xs, xt], 0)
        d2 = torch.cdist(x, x, p=2).pow(2)
        if self.sigma > 0:
            two_s2 = 2.0 * self.sigma ** 2
        else:
            off = d2[~torch.eye(len(x), dtype=torch.bool, device=x.device)]
            two_s2 = float(off.detach().median().clamp_min(1e-6))
        k = torch.exp(-d2 / two_s2)
        n = xs.shape[0]
        return (k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()).clamp_min(0.0)


def dynamic_lambda(epoch: int, epochs: int) -> float:
    """Paper Sec. 4.1: lambda = 2 / (1 + exp(-10 * epoch/epochs)) - 1."""
    p = epoch / max(1, epochs)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ----------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------
class MinMax18:
    """
    Per-feature MinMax to [0,1] across all samples and time steps (Sec. 3.1).
    Fit set [UNSPECIFIED]: source-train + target-adaptation training windows.
    Fitting on source only is known to explode a HUST current feature, so the
    target training windows are included; the blind test cells never are.
    """

    def fit(self, *arrays: np.ndarray) -> "MinMax18":
        flat = np.concatenate([a.reshape(-1, a.shape[-2] * a.shape[-1]) if a.ndim == 4
                               else a.reshape(-1, a.shape[-1]) for a in arrays], 0)
        self.lo = flat.min(0)
        self.rng = np.where(flat.max(0) - self.lo > 1e-12, flat.max(0) - self.lo, 1.0)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        shape = a.shape
        f = a.reshape(shape[0], shape[1], -1)
        return ((f - self.lo) / self.rng).astype(np.float32)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _r2(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def evaluate(y: np.ndarray, pred: np.ndarray, cells: np.ndarray,
             window_size: int) -> Dict:
    """
    Per-cell metrics (paper Table 3) and their mean (paper headline), plus pooled.

    MAPE uses the cell's cycle life as denominator (Sec. 4.3). Cycle life is
    recovered as max(RUL in cell) + window_size, since RUL = EOL - current_cycle
    and a cell's first window ends at cycle `window_size`.
    """
    per = {}
    for c in np.unique(cells):
        m = cells == c
        yc, pc = y[m], pred[m]
        life = float(yc.max()) + window_size
        per[str(c)] = {
            "n": int(m.sum()),
            "cycle_life": life,
            "rmse": float(np.sqrt(((yc - pc) ** 2).mean())),
            "r2": _r2(yc, pc),
            "mape": float((np.abs(yc - pc) / life).mean() * 100.0),
        }
    vals = list(per.values())
    macro = {k: float(np.nanmean([v[k] for v in vals])) for k in ("rmse", "r2", "mape")}
    pooled = {"rmse": float(np.sqrt(((y - pred) ** 2).mean())), "r2": _r2(y, pred)}
    return {"paper_macro": macro, "pooled": pooled, "per_cell": per}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, use: str = "target",
            bs: int = 1024) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)
    out = []
    for i in range(0, len(X), bs):
        yT, yS, _, _ = model(torch.from_numpy(X[i:i + bs]).to(device))
        out.append((yT if use == "target" else yS).cpu().numpy().ravel())
    return np.concatenate(out)


def train_one(args, seed: int, data: Dict, device: str, input_dim: int) -> Tuple[np.ndarray, Dict]:
    set_seed(seed)
    model = HybridoNetAdapt(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_lstm_layers=args.lstm_layers, num_heads=args.num_heads,
        dropout=args.dropout, attn_timestep=args.attn_timestep,
        node_points=args.node_points, node_output=args.node_output,
        attn_residual_norm=args.attn_residual_norm, post_node_norm=args.post_node_norm,
        predictor_norm=args.predictor_norm, theta_mode=args.theta_mode,
    ).to(device)

    # Paper: AdamW, fixed LR. Weight decay [UNSPECIFIED] -> PyTorch AdamW default.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    mmd = GaussianMMD(args.mmd_sigma)

    if len(data["Xs"]) == 0 or len(data["Xt"]) == 0:
        raise ValueError(f"Empty dataset: len(Xs)={len(data['Xs'])}, len(Xt)={len(data['Xt'])}")

    g = torch.Generator().manual_seed(seed)
    ds_s = TensorDataset(torch.from_numpy(data["Xs"]), torch.from_numpy(data["ys"]).unsqueeze(1))
    ds_t = TensorDataset(torch.from_numpy(data["Xt"]), torch.from_numpy(data["yt"]).unsqueeze(1))
    # drop_last keeps BatchNorm from ever seeing a batch of one
    bs_s = min(args.batch_size, len(ds_s))
    drop_s = True if len(ds_s) >= args.batch_size else False
    dl_s = DataLoader(ds_s, batch_size=bs_s, shuffle=True, drop_last=drop_s, generator=g)

    bs_t = min(args.batch_size, len(ds_t))
    drop_t = True if len(ds_t) >= args.batch_size else False
    dl_t = DataLoader(ds_t, batch_size=bs_t, shuffle=True, drop_last=drop_t, generator=g)

    best = (float("inf"), None, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = dynamic_lambda(epoch, args.epochs)
        it_t = iter(dl_t)
        tot = {"src": 0.0, "tgt": 0.0, "mmd": 0.0, "n": 0}
        for xs, ys in dl_s:
            try:
                xt, yt = next(it_t)
            except StopIteration:
                it_t = iter(dl_t)
                xt, yt = next(it_t)
            xs, ys, xt, yt = xs.to(device), ys.to(device), xt.to(device), yt.to(device)

            _, yS_s, _, z_s = model(xs)        # Eq. (2): source head alone
            yT_t, _, _, z_t = model(xt)        # Eq. (1): theta-blend
            l_src = mse(yS_s, ys)
            l_tgt = mse(yT_t, yt)
            l_mmd = mmd(z_s, z_t)
            loss = l_src + l_tgt + lam * l_mmd

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["src"] += l_src.item(); tot["tgt"] += l_tgt.item()
            tot["mmd"] += l_mmd.item(); tot["n"] += 1

        # validation: lowest RMSE selects the checkpoint (Sec. 4.1)
        vp = predict(model, data["Xv"], device, use=data["val_use"]) * data["y_max"]
        v_rmse = float(np.sqrt(((vp - data["yv_raw"]) ** 2).mean()))
        n = max(1, tot["n"])
        log.info(f"  seed {seed} ep {epoch:02d}/{args.epochs} | src {tot['src']/n:.4f} "
                 f"tgt {tot['tgt']/n:.4f} mmd {tot['mmd']/n:.4f} (lam {lam:.3f}) | "
                 f"val RMSE {v_rmse:7.2f} | val pred [{vp.min():.0f}, {vp.max():.0f}] | "
                 f"theta_S {model.theta_s.item():.3f} theta_T {model.theta_t.item():.3f}")
        if v_rmse < best[0]:
            best = (v_rmse, copy.deepcopy(model.state_dict()), epoch)

    if best[1] is not None:
        model.load_state_dict(best[1])
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({"state_dict": best[1], "best_epoch": best[2], "best_val_rmse": best[0],
                        "input_dim": input_dim, "args": vars(args)},
                       os.path.join(args.checkpoint_dir, f"paper_seed{seed}.pt"))
    else:
        log.warning(f"No checkpoint improved over initial val_rmse (val_rmse may have been NaN)")
    pred = predict(model, data["Xte"], device, use="target") * data["y_max"]
    return pred, {"seed": seed, "best_epoch": best[2], "best_val_rmse": best[0],
                  "theta_s": float(model.theta_s.item()), "theta_t": float(model.theta_t.item())}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HybridoNet-Adapt paper-faithful replication")
    ap.add_argument("--source", required=True, help="MATR/TRI .npz (X, Y, cell_ids)")
    ap.add_argument("--target", required=True, help="HUST/LHP .npz (X, Y, cell_ids)")
    # --- protocol (paper) ---
    ap.add_argument("--source-cells", default="severson-train",
                    choices=["severson-train", "severson-all", "file"],
                    help="paper: TRI training set (41 cells) -> severson-train")
    ap.add_argument("--source-cells-file", default=None)
    ap.add_argument("--val-frac", type=float, default=0.10, help="paper: 10%% validation")
    ap.add_argument("--val-domain", default="target", choices=["target", "source"],
                    help="[UNSPECIFIED] which training data the 10%% validation comes from. "
                         "target: select on target cells with Eq.(1); source: on source cells with Eq.(2)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10, help="paper: 10")
    ap.add_argument("--batch-size", type=int, default=128, help="paper: 128")
    ap.add_argument("--lr", type=float, default=5e-4, help="paper: 0.0005, fixed")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="[UNSPECIFIED] paper says AdamW; 0.01 is the PyTorch AdamW default")
    ap.add_argument("--num-runs", type=int, default=10, help="paper: 10, predictions averaged")
    ap.add_argument("--seed", type=int, default=0, help="first seed; runs use seed..seed+N-1")
    ap.add_argument("--window-size", type=int, default=30,
                    help="window length used in preprocessing (cycle life = max RUL + this)")
    # --- architecture (paper) ---
    ap.add_argument("--hidden-dim", type=int, default=64, help="paper: 64")
    ap.add_argument("--lstm-layers", type=int, default=2, help="paper: 2")
    ap.add_argument("--dropout", type=float, default=0.1, help="paper: 0.1")
    ap.add_argument("--attn-timestep", type=int, default=-2, help="paper: second-to-last (-2)")
    ap.add_argument("--node-points", type=int, default=2, help="paper: 2")
    ap.add_argument("--node-output", default="concat", choices=["concat", "last"],
                    help="INFERENCE: concat of h(0),h(1) gives the paper's 128-D predictor input")
    ap.add_argument("--predictor-norm", default="batchnorm", choices=["batchnorm", "layernorm", "none"],
                    help="paper: batchnorm")
    ap.add_argument("--theta-mode", default="free", choices=["free", "softmax"],
                    help="paper: free learnable scalars")
    # --- unspecified in paper ---
    ap.add_argument("--num-heads", type=int, default=4, help="[UNSPECIFIED]")
    ap.add_argument("--mmd-sigma", type=float, default=-1.0,
                    help="[UNSPECIFIED] <=0 uses the median heuristic")
    ap.add_argument("--attn-residual-norm", action="store_true",
                    help="[UNSPECIFIED] add residual+LayerNorm around attention (not in paper)")
    ap.add_argument("--post-node-norm", action="store_true",
                    help="[UNSPECIFIED] add LayerNorm after NODE (not in paper)")
    # --- io ---
    ap.add_argument("--checkpoint-dir", default="checkpoints/paper")
    ap.add_argument("--results-json", default="results/paper_replication.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    source_path = resolve_file_path(args.source)
    target_path = resolve_file_path(args.target)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {args.source} (resolved: {source_path})")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {args.target} (resolved: {target_path})")
    log.info(f"loading source from: {source_path}")
    log.info(f"loading target from: {target_path}")

    s = np.load(source_path, allow_pickle=True)
    t = np.load(target_path, allow_pickle=True)
    Xs_all, Ys_all, cs_all = s["X"].astype(np.float32), s["Y"].astype(np.float32), s["cell_ids"].astype(str)
    Xt_all, Yt_all, ct_all = t["X"].astype(np.float32), t["Y"].astype(np.float32), t["cell_ids"].astype(str)
    input_dim = int(np.prod(Xs_all.shape[2:]))

    # ---- target split: paper's 22 test cells vs the 55 training cells ----
    chan = np.array([hust_channel(c) for c in ct_all])
    is_test = np.isin(chan, list(PAPER_TEST_CELLS))
    found = sorted(set(chan[is_test]))
    missing = sorted(set(PAPER_TEST_CELLS) - set(found))
    if missing:
        raise SystemExit(f"Paper test channels not found in target cell ids: {missing}. "
                         f"Example target id: {ct_all[0]!r}")
    tgt_train_cells = np.unique(ct_all[~is_test])
    log.info(f"target: {len(found)} paper test cells, {len(tgt_train_cells)} training cells "
             f"(paper: 22 / 55)")

    # ---- source cells ----
    src_cells = select_source_cells(cs_all, args.source_cells, args.source_cells_file)
    log.info(f"source: {len(src_cells)} cells via '{args.source_cells}' (paper: 41)")

    # ---- 90/10 validation split (cell level) ----
    if args.val_domain == "target":
        tgt_adapt, val_cells = split_cells(tgt_train_cells, args.val_frac, args.split_seed)
        src_train = src_cells
        Xv = Xt_all[np.isin(ct_all, val_cells)]; yv = Yt_all[np.isin(ct_all, val_cells)]
        val_use = "target"
    else:
        src_train, val_cells = split_cells(src_cells, args.val_frac, args.split_seed)
        tgt_adapt = tgt_train_cells
        Xv = Xs_all[np.isin(cs_all, val_cells)]; yv = Ys_all[np.isin(cs_all, val_cells)]
        val_use = "source"

    ms, mt = np.isin(cs_all, src_train), np.isin(ct_all, tgt_adapt)
    Xs, ys, Xt, yt = Xs_all[ms], Ys_all[ms], Xt_all[mt], Yt_all[mt]
    Xte, yte, cte = Xt_all[is_test], Yt_all[is_test], ct_all[is_test]
    log.info(f"windows | source train {len(Xs)} ({len(src_train)} cells) | target adapt {len(Xt)} "
             f"({len(tgt_adapt)} cells) | val {len(Xv)} ({len(val_cells)} {args.val_domain} cells) | "
             f"test {len(Xte)} (22 cells)")

    # ---- scaling (fit on training windows only) ----
    sc = MinMax18().fit(Xs, Xt)
    y_max = float(max(ys.max(), yt.max()))     # label scale for the sigmoid output
    data = {
        "Xs": sc.transform(Xs), "ys": (ys / y_max).astype(np.float32),
        "Xt": sc.transform(Xt), "yt": (yt / y_max).astype(np.float32),
        "Xv": sc.transform(Xv), "yv_raw": yv, "val_use": val_use,
        "Xte": sc.transform(Xte), "y_max": y_max,
    }
    oob = (data["Xte"] < -0.5) | (data["Xte"] > 1.5)
    if oob.any():
        log.warning(f"{oob.mean()*100:.2f}% of test feature values fall outside [-0.5, 1.5] "
                    f"after scaling — check the feature scaler.")
    log.info(f"label scale y_max = {y_max:.0f} cycles")

    # ---- 10 runs, ensemble average (Sec. 4.1) ----
    preds, runs = [], []
    for i in range(args.num_runs):
        seed = args.seed + i
        log.info(f"===== run {i+1}/{args.num_runs} (seed {seed}) =====")
        p, info = train_one(args, seed, data, device, input_dim)
        ev = evaluate(yte, p, cte, args.window_size)
        info.update({"paper_macro": ev["paper_macro"], "pooled": ev["pooled"]})
        log.info(f"  run {i+1}: per-cell mean RMSE {ev['paper_macro']['rmse']:.2f} | "
                 f"R2 {ev['paper_macro']['r2']:.3f} | MAPE {ev['paper_macro']['mape']:.2f}% "
                 f"|| pooled RMSE {ev['pooled']['rmse']:.2f}  R2 {ev['pooled']['r2']:.3f} "
                 f"| epoch {info['best_epoch']}")
        preds.append(p); runs.append(info)

    ens = evaluate(yte, np.mean(preds, 0), cte, args.window_size)
    em, ep = ens["paper_macro"], ens["pooled"]

    print("\n" + "=" * 78)
    print(f"PAPER REPLICATION — Table 2 'All' setting, {args.num_runs}-run ensemble")
    print("=" * 78)
    print(f"{'metric':<34}{'paper':>10}{'ours':>12}{'diff':>12}")
    for k, lab in (("rmse", "RMSE (mean of per-cell)"), ("r2", "R2 (mean of per-cell)"),
                   ("mape", "MAPE % (mean of per-cell)")):
        pv, ov = PAPER_HEADLINE[k], em[k]
        rel = (ov - pv) / pv * 100
        print(f"{lab:<34}{pv:>10.2f}{ov:>12.2f}{rel:>+11.1f}%")
    print(f"{'pooled RMSE (all windows)':<34}{'—':>10}{ep['rmse']:>12.2f}")
    print(f"{'pooled R2 (all windows)':<34}{'—':>10}{ep['r2']:>12.3f}")

    print("\nPer-cell comparison with Table 3 (HybridoNet-Adapt column)")
    print(f"{'channel':<9}{'paper RMSE':>11}{'ours':>9}   {'paper R2':>9}{'ours':>7}   "
          f"{'paper MAPE':>11}{'ours':>8}")
    by_chan = {hust_channel(c): v for c, v in ens["per_cell"].items() if hust_channel(c) is not None}
    for ch, (pr, p2, pm) in PAPER_TEST_CELLS.items():
        if ch in by_chan:
            o = by_chan[ch]
            print(f"{ch:<9}{pr:>11.1f}{o['rmse']:>9.1f}   {p2:>9.2f}{o['r2']:>7.2f}   "
                  f"{pm:>10.2f}%{o['mape']:>7.2f}%")
        else:
            print(f"{ch:<9}{pr:>11.1f}{'N/A':>9}   {p2:>9.2f}{'N/A':>7}   "
                  f"{pm:>10.2f}%{'N/A':>8}")

    rm = np.array([r["paper_macro"]["rmse"] for r in runs])
    print(f"\nIndividual runs: per-cell mean RMSE {rm.mean():.2f} ± {rm.std():.2f}")
    print("=" * 78)

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump({"args": vars(args), "paper": PAPER_HEADLINE,
                   "ensemble": {"paper_macro": em, "pooled": ep, "per_cell": ens["per_cell"]},
                   "runs": runs,
                   "split": {"source_cells": src_train.tolist(), "target_adapt": tgt_adapt.tolist(),
                             "val_cells": val_cells.tolist(), "test_cells": found}},
                  f, indent=2)
    log.info(f"saved {args.results_json}")


if __name__ == "__main__":
    main()
