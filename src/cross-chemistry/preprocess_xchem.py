#!/usr/bin/env python3
"""
HybridoNet Cross-Chemistry Preprocessing Pipeline (Stage 0).

Extends the HybridoNet-Adapt feature pipeline with window-relative differential
features (rate information) to address model mean-reversion in cross-domain transfer.

Key methodology:
1. Signal Filtering: 1D median filter (kernel=3) on Voltage (V), Current (I), and Capacity (Q) per cycle.
2. Statistical Extraction: 6 features (Mean, Std, Min, Max, Var, Median) for each of the 3 signals -> 18 base features/cycle.
3. Differential Rate Features:
   delta = cell_tensor - cell_tensor[0:1, :, :]
   Computes window-relative deltas for selected signals (e.g. Q and I), dropping unselected signals.
   - Arm A: 18 base features (no deltas)
   - Arm B: 30 features (18 base + 12 deltas for Q, I)
   - Arm C: 36 features (18 base + 18 deltas for V, I, Q)
4. Target Formulation: RUL = EOL - current_cycle.
5. Self-Describing Output: saves X, Y, cell_ids, sample_ids, feature_names, and num_features to .npz.
"""

import os
import sys
import argparse
import logging
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from scipy.signal import medfilt
except ImportError:
    def medfilt(x, kernel_size=3):
        k = kernel_size // 2
        pad_x = np.pad(x, (k, k), mode="edge")
        out = np.zeros_like(x)
        for i in range(len(x)):
            out[i] = np.median(pad_x[i:i + kernel_size])
        return out

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [XChemPreprocess] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XChemPreprocess")

CANONICAL_SIGNALS = ["V", "I", "Q"]
SIGNAL_INDEX = {"V": 0, "I": 1, "Q": 2}
STAT_NAMES = ["mean", "std", "min", "max", "var", "med"]

BASE_FEATURE_NAMES_18 = [
    f"{s}_{st}" for s in CANONICAL_SIGNALS for st in STAT_NAMES
]


def build_feature_names(use_deltas: bool, active_delta_signals: List[str]) -> List[str]:
    """Generates ordered feature names matching the tensor column layout."""
    names = list(BASE_FEATURE_NAMES_18)
    if use_deltas:
        for s in active_delta_signals:
            for st in STAT_NAMES:
                names.append(f"d{s}_{st}")
    return names


def parse_delta_signals(delta_signals_arg: str) -> List[str]:
    """Parses and canonicalizes the list of signals to compute deltas for."""
    if not delta_signals_arg:
        return []
    raw_tokens = [t.strip().upper() for t in delta_signals_arg.split(",") if t.strip()]
    if "ALL" in raw_tokens:
        return list(CANONICAL_SIGNALS)
    active = [s for s in CANONICAL_SIGNALS if s in raw_tokens]
    return active


def compute_cycle_statistics(
    voltage: np.ndarray,
    current: np.ndarray,
    capacity: np.ndarray,
    filter_kernel: int = 3
) -> np.ndarray:
    """
    Computes 6 statistical features for Voltage, Current, Capacity after median filtering.
    Features: [Mean, Std, Min, Max, Variance, Median]

    Returns:
        feature_matrix: Shape (3, 6) -> 18 features
    """
    signals = [voltage, current, capacity]
    feature_matrix = np.zeros((3, 6), dtype=np.float32)

    for i, sig in enumerate(signals):
        if len(sig) == 0:
            continue
        clean_sig = medfilt(sig, kernel_size=filter_kernel)
        mean_val = float(np.mean(clean_sig))
        std_val = float(np.std(clean_sig))
        min_val = float(np.min(clean_sig))
        max_val = float(np.max(clean_sig))
        var_val = float(np.var(clean_sig))
        med_val = float(np.median(clean_sig))
        feature_matrix[i] = [mean_val, std_val, min_val, max_val, var_val, med_val]

    return feature_matrix


def extract_window_tensor(
    cycle_data: Dict[int, Dict[str, np.ndarray]],
    window_cycles: List[int],
    num_samples: int = 10,
    use_deltas: bool = True,
    active_delta_signals: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Uniformly samples `num_samples` (10) cycles from a list of window cycles.
    If use_deltas is True, appends rate deltas relative to cycle 0 in the window
    for each signal in active_delta_signals (dropping unselected signals).

    Returns:
        tensor:
          - (10, 18) if use_deltas is False
          - (10, 18 + 6 * len(active_delta_signals)) if use_deltas is True
            (e.g. 10x30 for Q,I deltas, 10x36 for all deltas)
    """
    if len(window_cycles) < num_samples:
        return None

    # Uniform 10-cycle sampling across the observation window
    idx_uniform = np.linspace(0, len(window_cycles) - 1, num_samples, dtype=int)
    selected_cycles = [window_cycles[i] for i in idx_uniform]

    cell_tensor = np.zeros((num_samples, 3, 6), dtype=np.float32)
    for step_idx, cyc in enumerate(selected_cycles):
        c_dict = cycle_data[cyc]
        v = np.array(c_dict.get("voltage", c_dict.get("V", [])))
        i = np.array(c_dict.get("current", c_dict.get("I", [])))
        q = np.array(c_dict.get("capacity", c_dict.get("Q", c_dict.get("Qd", []))))
        feat_3x6 = compute_cycle_statistics(v, i, q)
        cell_tensor[step_idx] = feat_3x6

    # Flatten base features: (10, 3, 6) -> (10, 18)
    base_flat = cell_tensor.reshape(num_samples, 18)

    if not use_deltas or not active_delta_signals:
        return base_flat

    # Delta relative to the first sampled cycle of the observation window
    # delta shape: (10, 3, 6)
    delta_tensor = cell_tensor - cell_tensor[0:1, :, :]

    # Extract deltas only for active signals (unselected signals are dropped, not zero-padded)
    delta_blocks = []
    for s in active_delta_signals:
        sig_idx = SIGNAL_INDEX[s]
        delta_blocks.append(delta_tensor[:, sig_idx, :])  # (10, 6)

    delta_flat = np.concatenate(delta_blocks, axis=1)  # (10, 6 * num_active)
    out = np.concatenate([base_flat, delta_flat], axis=1)  # (10, 18 + 6 * num_active)
    return out


def extract_cell_samples(
    cycle_data: Dict[int, Dict[str, np.ndarray]],
    eol: float,
    window_size: int = 30,
    stride: int = 30,
    num_samples: int = 10,
    rolling: bool = True,
    use_deltas: bool = True,
    active_delta_signals: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """Extracts window tensors and true RUL targets (RUL = EOL - current_cycle)."""
    available_cycles = sorted([c for c in cycle_data.keys() if c > 0])
    if len(available_cycles) < window_size:
        return [], []

    samples = []
    ruls = []

    if not rolling:
        window = [c for c in available_cycles if c <= window_size]
        if len(window) >= num_samples:
            tensor = extract_window_tensor(
                cycle_data, window, num_samples=num_samples,
                use_deltas=use_deltas, active_delta_signals=active_delta_signals
            )
            if tensor is not None and not np.isnan(tensor).any():
                samples.append(tensor)
                ruls.append(max(0.0, float(eol - window[-1])))
        return samples, ruls

    for end_idx in range(window_size, len(available_cycles) + 1, stride):
        window = available_cycles[end_idx - window_size:end_idx]
        current_cycle = window[-1]

        if current_cycle >= eol:
            break

        tensor = extract_window_tensor(
            cycle_data, window, num_samples=num_samples,
            use_deltas=use_deltas, active_delta_signals=active_delta_signals
        )
        if tensor is not None and not np.isnan(tensor).any():
            true_rul = float(eol - current_cycle)
            samples.append(tensor)
            ruls.append(true_rul)

    return samples, ruls


def process_parquet_dataset(
    parquet_path: str,
    domain_name: str,
    window_size: int = 30,
    stride: int = 10,
    num_samples: int = 10,
    rolling: bool = True,
    use_deltas: bool = True,
    active_delta_signals: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Processes standardized battery parquets and extracts feature tensors + RUL labels + cell IDs."""
    num_feat = 18 + (6 * len(active_delta_signals) if (use_deltas and active_delta_signals) else 0)

    if not os.path.exists(parquet_path):
        logger.warning(f"File not found: {parquet_path}")
        return np.empty((0, num_samples, num_feat)), np.empty((0,)), [], []

    df = pd.read_parquet(parquet_path)
    cell_unique = df["cell_id"].unique()

    all_tensors = []
    all_ruls = []
    all_sample_ids = []
    all_cell_ids = []

    for cid in cell_unique:
        cell_df = df[df["cell_id"] == cid]

        # Determine cell End of Life (EOL)
        if "cycle_life" in cell_df.columns and not cell_df["cycle_life"].isna().all():
            eol = float(cell_df["cycle_life"].dropna().iloc[0])
        elif "max_cycle" in cell_df.columns and not cell_df["max_cycle"].isna().all():
            eol = float(cell_df["max_cycle"].dropna().iloc[0])
        else:
            eol = float(cell_df["cycle_number"].max())
            logger.warning(
                f"⚠️ [EOL Fallback] Cell '{cid}' in domain '{domain_name}' ({parquet_path}) is missing "
                f"'cycle_life' and 'max_cycle' columns! Falling back to cycle_number.max() ({eol:.0f} cyc). "
                f"This may distort RUL targets near EOL if the cell test ended before capacity threshold failure."
            )

        cycle_data = {}
        for cyc_num, group in cell_df.groupby("cycle_number"):
            def get_col(df_grp, candidates):
                for c in candidates:
                    if c in df_grp.columns:
                        return df_grp[c].values
                return np.array([])

            v = get_col(group, ["voltage", "voltage_V", "Voltage", "V", "voltage_v"])
            i = get_col(group, ["current", "current_A", "Current", "I", "current_a"])
            q = get_col(group, ["discharge_capacity", "capacity", "capacity_Ah", "Discharge_Capacity", "Q", "Capacity"])
            cycle_data[int(cyc_num)] = {"voltage": v, "current": i, "capacity": q}

        tensors, ruls = extract_cell_samples(
            cycle_data, eol, window_size=window_size, stride=stride, num_samples=num_samples,
            rolling=rolling, use_deltas=use_deltas, active_delta_signals=active_delta_signals
        )

        cell_global_id = f"{domain_name}_{cid}"
        for s_idx, (t_mat, r_val) in enumerate(zip(tensors, ruls)):
            all_tensors.append(t_mat)
            all_ruls.append(r_val)
            all_sample_ids.append(f"{cell_global_id}_w{s_idx}")
            all_cell_ids.append(cell_global_id)

    if len(all_tensors) == 0:
        return np.empty((0, num_samples, num_feat)), np.empty((0,)), [], []

    return np.array(all_tensors, dtype=np.float32), np.array(all_ruls, dtype=np.float32), all_sample_ids, all_cell_ids


def process_domain_parquet_files(
    file_list: List[str],
    domain_name: str,
    window_size: int = 30,
    stride: int = 10,
    num_samples: int = 10,
    rolling: bool = True,
    use_deltas: bool = True,
    active_delta_signals: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Processes and bundles multiple parquet files belonging to the same domain."""
    num_feat = 18 + (6 * len(active_delta_signals) if (use_deltas and active_delta_signals) else 0)
    all_X, all_Y, all_samples, all_cells = [], [], [], []

    for p_file in file_list:
        X, Y, sample_ids, cell_ids = process_parquet_dataset(
            p_file, domain_name, window_size=window_size, stride=stride, num_samples=num_samples,
            rolling=rolling, use_deltas=use_deltas, active_delta_signals=active_delta_signals
        )
        if len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)
            all_samples.extend(sample_ids)
            all_cells.extend(cell_ids)

    if len(all_X) == 0:
        return np.empty((0, num_samples, num_feat)), np.empty((0,)), [], []

    return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0), all_samples, all_cells


CANONICAL_ALIASES = {
    "tri": ["TRI", "MATR"],
    "matr": ["TRI", "MATR"],
    "lhp": ["HUST", "LHP"],
    "hust": ["HUST", "LHP"],
}

DEFAULT_DATA_DIR = "data/hybridonet/raw"
FALLBACK_DATA_DIR = "data/real_processed"


def has_parquet_files(path: str) -> bool:
    """Checks if a directory or its immediate subdirectories contain any .parquet files."""
    if not os.path.exists(path):
        return False
    if glob.glob(os.path.join(path, "*.parquet")):
        return True
    try:
        for d in os.listdir(path):
            sub = os.path.join(path, d)
            if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.parquet")):
                return True
    except Exception:
        pass
    return False


def save_feature_dataset(
    output_dir: str,
    domain_name: str,
    X: np.ndarray,
    Y: np.ndarray,
    sample_ids: List[str],
    cell_ids: List[str],
    feature_names: List[str]
) -> List[str]:
    """
    Saves raw unscaled feature tensors and metadata to .npz archives, writing canonical aliases.
    Records feature_names and num_features for self-describing archives.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    num_features = int(np.prod(X.shape[2:])) if len(X) > 0 else len(feature_names)

    primary_file = os.path.join(output_dir, f"{domain_name}_raw_features.npz")
    np.savez_compressed(
        primary_file,
        X=X,
        Y=Y,
        sample_ids=np.array(sample_ids),
        cell_ids=np.array(cell_ids),
        feature_names=np.array(feature_names),
        num_features=num_features
    )
    saved_paths.append(primary_file)

    dom_lower = domain_name.lower()
    if dom_lower in CANONICAL_ALIASES:
        for alias in CANONICAL_ALIASES[dom_lower]:
            alias_file = os.path.join(output_dir, f"{alias}_raw_features.npz")
            if os.path.abspath(alias_file) != os.path.abspath(primary_file):
                np.savez_compressed(
                    alias_file,
                    X=X,
                    Y=Y,
                    sample_ids=np.array(sample_ids),
                    cell_ids=np.array(cell_ids),
                    feature_names=np.array(feature_names),
                    num_features=num_features
                )
                saved_paths.append(alias_file)
                logger.info(f"Saved canonical alias -> {alias_file}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Cross-Chemistry Battery Feature Preprocessing (Stage 0)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing battery parquets")
    parser.add_argument("--output-dir", type=str, default="data/xchem/processed", help="Output directory for processed .npz files")
    parser.add_argument("--window-size", type=int, default=30, help="Observation window size (cycles, default: 30)")
    parser.add_argument("--stride", type=int, default=10, help="Window stride for rolling RUL samples (default: 10)")
    parser.add_argument("--num-samples", type=int, default=10, help="Uniformly sampled cycles per window (default: 10)")
    parser.add_argument("--early-only", action="store_true", help="Extract only the first early window per cell")
    parser.add_argument("--use-deltas", dest="use_deltas", action="store_true", default=True, help="Append window-relative delta rate features (default: True)")
    parser.add_argument("--no-use-deltas", dest="use_deltas", action="store_false", help="Disable delta rate features (reproduce 18-D baseline)")
    parser.add_argument("--delta-signals", type=str, default="Q,I", help="Comma-separated signals to compute deltas for (default: 'Q,I'; use 'V,I,Q' or 'all' for all deltas)")
    args = parser.parse_args()

    active_delta_signals = parse_delta_signals(args.delta_signals) if args.use_deltas else []
    feature_names = build_feature_names(args.use_deltas, active_delta_signals)
    num_features = len(feature_names)

    os.makedirs(args.output_dir, exist_ok=True)
    rolling = not args.early_only

    data_dir = args.data_dir
    if data_dir == DEFAULT_DATA_DIR and not has_parquet_files(data_dir):
        if has_parquet_files(FALLBACK_DATA_DIR):
            logger.info(f"No parquet datasets in '{data_dir}'; automatically using '{FALLBACK_DATA_DIR}'")
            data_dir = FALLBACK_DATA_DIR

    logger.info(
        f"Extracting features from '{data_dir}' (Deltas: {args.use_deltas}, "
        f"Delta Signals: {active_delta_signals}, Feature Dim: {num_features}-D)..."
    )

    domains_processed = set()

    # 1. Discover domain subdirectories
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for d in subdirs:
            domain_path = os.path.join(data_dir, d)
            parquets = glob.glob(os.path.join(domain_path, "*.parquet"))
            if parquets:
                X, Y, s_ids, c_ids = process_domain_parquet_files(
                    parquets, d, window_size=args.window_size, stride=args.stride,
                    num_samples=args.num_samples, rolling=rolling,
                    use_deltas=args.use_deltas, active_delta_signals=active_delta_signals
                )
                if len(X) > 0:
                    saved = save_feature_dataset(args.output_dir, d, X, Y, s_ids, c_ids, feature_names)
                    n_cells = len(np.unique(c_ids))
                    logger.info(
                        f"Saved {d}: {len(X)} samples across {n_cells} cells, "
                        f"shape {X.shape}, RUL: [{Y.min():.0f}, {Y.max():.0f}] cyc -> {saved[0]}"
                    )
                    domains_processed.add(d)

    # 2. Discover top-level parquet files
    if os.path.exists(data_dir):
        top_parquets = glob.glob(os.path.join(data_dir, "*.parquet"))
        for p_file in top_parquets:
            domain = os.path.splitext(os.path.basename(p_file))[0]
            if domain not in domains_processed:
                X, Y, s_ids, c_ids = process_parquet_dataset(
                    p_file, domain, window_size=args.window_size, stride=args.stride,
                    num_samples=args.num_samples, rolling=rolling,
                    use_deltas=args.use_deltas, active_delta_signals=active_delta_signals
                )
                if len(X) > 0:
                    saved = save_feature_dataset(args.output_dir, domain, X, Y, s_ids, c_ids, feature_names)
                    n_cells = len(np.unique(c_ids))
                    logger.info(
                        f"Saved {domain}: {len(X)} samples across {n_cells} cells, "
                        f"shape {X.shape}, RUL: [{Y.min():.0f}, {Y.max():.0f}] cyc -> {saved[0]}"
                    )
                    domains_processed.add(domain)

    if not domains_processed:
        logger.warning(f"No parquet datasets found in '{data_dir}'.")
    else:
        logger.info(f"Preprocessing completed: {num_features}-D tensors written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
