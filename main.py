#!/usr/bin/env python3
"""
Master End-to-End Orchestration Pipeline for Lithium-Ion Battery SOH & RUL Estimation.
Executes Phase 1 (Data Acquisition), Phase 2 (Feature Engineering), Phase 3 (Modeling & Hypothesis Testing),
and Phase 4 (Validation Figures & Electrochemical Research Report).
"""

import os
import sys
import logging
import argparse
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MAIN] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainPipeline")


def run_step(name: str, cmd: list):
    logger.info("="*70)
    logger.info(f"STARTING -> {name}")
    logger.info("="*70)
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        logger.error(f"Step '{name}' failed with return code {ret.returncode}.")
        sys.exit(ret.returncode)
    logger.info(f"COMPLETED -> {name}\n")


def main():
    parser = argparse.ArgumentParser(description="Lithium-Ion Battery SOH & RUL ML Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip Phase 1 (Data Acquisition) if already cached")
    parser.add_argument("--skip-features", action="store_true", help="Skip Phase 2 (Feature Engineering) if already featurized")
    parser.add_argument("--only-phase-3-4", action="store_true", help="Run only Phase 3 (Modeling) and Phase 4 (Figures)")
    args = parser.parse_args()

    python_exe = sys.executable

    # Phase 1: Data Acquisition
    if not (args.skip_data or args.only_phase_3_4):
        run_step("Phase 1: Data Acquisition (124 Cells Stanford/MIT Dataset)", [python_exe, "src/data_acquisition.py"])
    else:
        logger.info("Skipping Phase 1: Data Acquisition (using cached dataset)...")

    # Phase 2: Feature Engineering
    if not (args.skip_features or args.only_phase_3_4):
        run_step("Phase 2: Feature Engineering (dQ/dV & Delta Q Analysis)", [python_exe, "src/feature_engineering.py"])
    else:
        logger.info("Skipping Phase 2: Feature Engineering (using cached features)...")

    # Phase 3: Modeling & Hypothesis Testing
    run_step("Phase 3: Modeling & Hypothesis Testing (Baseline vs. Severson vs. Physics-Informed)", [python_exe, "src/modeling.py"])

    # Phase 4: Validation & Figures
    run_step("Phase 4: Publication Figures & Visualization Suite", [python_exe, "src/visualization.py"])

    logger.info("="*70)
    logger.info("END-TO-END PIPELINE EXECUTED SUCCESSFULLY!")
    logger.info("Check results/ for model evaluation metrics and figures/ for publication plots.")
    logger.info("="*70)


if __name__ == "__main__":
    main()
