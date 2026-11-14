# 2026-11-08 — Accuracy matrix: calibration × levels × distance

Purpose: map how calibration strategy, number of quantization levels, and distance function
affect matching accuracy, so that methods are compared inside one design space
(the 8-bin thermometer = quantile, 8 levels, idx_L1; int8 ≈ shared uniform, 256 levels, dot).
Accuracy only (environment-independent); speed is measured separately.

## Fixed
- Regenerated 2026-10-14 XFeat cache (`CSE527/data/2026-10-14_regen_cache`), max side 640, top-k 1000.
- Calibration: the same 191,581 COCO descriptors (192 images). Reconstruction value of a bin =
  calibration conditional mean of that bin, for every strategy.
- Code rule `count(x > edge)`, exact MNN, smallest index on ties, pooled precision at 1/3/5 px.

## Changed
- Strategy (per dimension unless noted): `uniform` (min–max), `percentile` (0.1–99.9 % clip, equal width),
  `quantile` (equal frequency), `lloyd` (1D Lloyd–Max, init = quantile), `uniform_shared`, `quantile_shared`
  (one set of edges for all 64 dimensions).
- Levels: 4, 8, 16, 32, 256 (packed 16, 24, 32, 40, 64 B per descriptor).
- Distance: `idx_L1` (= thermometer Hamming), `idx_L2` (integer bin index), `recon_cos` (cosine of reconstructed values).

## Implementation checks
- `idx_L1` uses the verified C SAD kernel from `../2026-11-08_fair_speed/speed_kernels.py`; `idx_L2` uses a float32 GEMM
  on integer codes (all partial sums < 2^24, therefore exact). For the first pair of each dataset both are checked
  against NumPy integer references for all 30 calibrations.
- Recomputed quantile-8 edges differ from the saved 2026-10-14 edges by ≤1.3e-7; quantile-8 results reproduce
  the earlier numbers (HPatches idx_L1 58.51 %, idx_L2 59.46 %, recon_cos 59.55 %).

## Data status
Validation only: COCO synthetic (192 pairs) and HPatches (580 pairs). Any configuration chosen from this table must be
confirmed on the reserved final test set (MegaDepth-1500), which is not used here.

## Files
`accuracy_matrix.py`; `results/summary.csv`, `results/by_type.csv`, `results/per_pair.csv.gz`,
`results/calibration_fit.csv` (code entropy, calibration reconstruction MSE, empty bins), `results/edges_all.npz`,
`results/environment.json`, `results/run.log`.
Run: `python accuracy_matrix.py CACHE ORIG_CODE ../2026-11-08_fair_speed OUT` (≈7 min on 2 vCPU).
