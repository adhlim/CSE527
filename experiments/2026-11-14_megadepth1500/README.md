# 2026-11-14 — MegaDepth-1500 held-out evaluation (prepared, not run)

Purpose: the one evaluation on data that were never used in this project. Protocol, configurations and predictions are
fixed in `paper/preregistration_megadepth1500.md`. This folder is the code that the pre-registration refers to.

Status: code written and checked locally as far as possible; **the evaluation itself has not been run.**

## Files
- `frozen_configs.npz` — every parameter of the 15 configurations (sha256 8666ae9e8db35dab). Built by `build_frozen.py`.
- `md_configs.py` — the 15 configurations as NumPy functions. `md_pose.py` — pose estimation and AUC.
- `md_eval.py` — extraction, matching, pose, summary (Colab). `md_predictions.py` — judges P1–P10.
- `2026-11-14_megadepth1500.ipynb` — Colab runner. `inputs/` — pair list from the XFeat repository, verbatim `features.py`.

## What was checked locally
- `md_configs.match` reproduces the validation precision of all 15 configurations on HPatches and COCO
  (30 of 30 equal to two decimals; `results/config_check_*.json`).
- `md_pose.py` on synthetic two-view data: exact poses from exact matches; median error 0.06° with 0.5 px noise and
  0.09° with 50 % outliers added; identical output for a repeated seed; AUC = 1 for perfect and 0 for failed poses.
- Matching all 15 configurations on one 4096 × 4096 pair takes about 3.3 s on 2 threads.

## What was NOT checked locally
Extraction on MegaDepth images, the dataset download, and the full loop of `md_eval.py` need PyTorch and the dataset
and have not been executed. Step 1 of the notebook (float only) is the first real execution.

## How to run
1. Upload this folder to `MyDrive/Colab Notebooks/2026-11-14_megadepth1500/`, open the notebook, CPU runtime.
2. Step 1: float-only check. Setting B / C0 should be near the published 42.6 / 56.4 / 67.7.
3. Commit the pre-registration and this folder to git; paste the commit hash into `PREREG_COMMIT`.
4. Step 2: the registered run (resumable). Step 3: `md_predictions.py`.
Outputs: `results_colab/pairs_<setting>_<config>.csv`, `summary.csv`, `predictions.csv`, `manifest.json`.
