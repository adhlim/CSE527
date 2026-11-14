# 2026-11-14 — Dimensionality reduction (PCA, GFR) vs scalar quantization at equal bytes

Purpose: answer the advisor's question "how does the current approach compare with Gram-Schmidt Feature Reduction?"
with numbers. Three questions: (1) does GFR beat PCA on XFeat descriptors, (2) is reducing dimensions better than
reducing bits per dimension at the same byte budget, (3) does combining both help.
Validation only (COCO synthetic 192 pairs, HPatches 580 pairs). No setting is selected for the final test here.

## Fixed
- Regenerated 2026-10-14 XFeat cache (`data/2026-10-14_regen_cache`), max side 640, top-k 1000.
- Projections and quantization edges are fitted on the 191,581 COCO calibration descriptors only.
- Exact MNN, smallest index on ties, pooled precision at 1/3/5 px (same evaluation as `2026-11-08_accuracy_matrix`).

## Changed
- Projection: `pca` (mean-centred eigenvectors), `gfr` (see below), `truncate` (first m raw dimensions, naive baseline).
- Kept dimensions m: 48, 32, 24, 16.
- Reduced float descriptors: L2-normalised, cosine.
- Reduced + scalar quantization (pca, gfr only): per-dimension `percentile` (0.1–99.9 %) or `quantile` edges,
  4 / 8 / 16 / 256 levels, distance `idx_L2` (integer bin index) or `recon_cos` (cosine of calibration conditional means).
  Bytes = m × bits / 8.

## GFR implementation (`fit_projections.py`)
Algorithm 2 of Safavigerdini et al., "Gram-Schmidt Feature Reduction for Local Feature Descriptor Compression"
(CVPR Workshops 2026), with the function-space Gram–Schmidt of Yaghooti et al. (IEEE Trans. IT 2025):
at step j take the top eigenvector ν_j of the residual covariance, z_j = ν_jᵀX, orthonormalise the new degree-2
multilinear functions {z_j} ∪ {z_j z_i, i<j} against all previous ones (constant included; block Gram–Schmidt applied
twice, then QR inside the block), and subtract their projection from the residual data.
This is our reading of the paper; the authors' code was not used. Their extra choices (score normalisation variants,
alternative deflation rules) are not explored.

Checks (in `results/fit_info.json`):
- With degree 1 only, the loop returns the PCA subspace (‖P_gfr1 − P_pca‖ ≤ 1e-13 for m = 8…48).
- GFR directions are orthonormal (error 2e-9); no function was dropped as linearly dependent (1,177 functions for m = 48).
- A full 64-D PCA rotation leaves pairwise distances unchanged (asserted on the first pair).
- Re-running the 68 configurations of the first, smaller run reproduced identical numbers (`results/_old/`, kept because
  deleting is disabled in this workspace; safe to remove).

## Results (precision @3px, %)
Float 64-D cosine (256 B): HPatches 60.59, COCO 70.57.

Reduced float descriptors:
| m | bytes | HPatches PCA | HPatches GFR | HPatches truncate | COCO PCA | COCO GFR | COCO truncate |
|---|---|---|---|---|---|---|---|
| 48 | 192 | 60.61 | 60.60 | 59.81 | 70.74 | 70.71 | 70.13 |
| 32 | 128 | 59.91 | 59.93 | 58.51 | 70.51 | 70.55 | 68.86 |
| 24 | 96 | 58.06 | 58.07 | 56.34 | 68.80 | 68.84 | 66.87 |
| 16 | 64 | 53.18 | 53.25 | 51.10 | 64.01 | 64.20 | 62.00 |

Best configuration per family at equal bytes (HPatches / COCO):
| bytes | 64-D scalar quantization | reduce + quantize |
|---|---|---|
| 16 | 2 bit: 57.59 / 66.84 | 32-D × 4 bit: 59.11 / 69.61 |
| 24 | 3 bit: 59.55 / 69.34 | 48-D × 4 bit: 60.04 / 70.05 |
| 32 | 4 bit: 60.39 / 70.29 | 32-D × 8 bit: 59.92 / 70.53 |
| 48 | (5 bit = 40 B: 60.54 / 70.47) | 48-D × 8 bit: 60.61 / 70.71 |
| 64 | 8 bit: 60.59 / 70.59 | — |

## Observations
1. GFR ≈ PCA on XFeat: over 68 matched configurations GFR − PCA is −0.006 pp on HPatches (range −0.34…+0.20) and
   −0.07 pp on COCO. The subspaces differ by 3–14° (largest principal angle) for m ≤ 32; at m = 48 one low-variance direction differs completely (89°) without a visible effect on accuracy. This agrees with the paper's own
   report that gains over PCA deflation are not significant and are smaller for learned descriptors.
2. Dimension reduction alone is dominated by scalar quantization: 32-D float (128 B) gives 59.9 on HPatches, below
   64-D × 4 bit (32 B, 60.39).
3. The combination helps at small budgets: at 16 B, 32-D × 4 bit beats 64-D × 2 bit by 1.5 pp (HPatches) and 2.8 pp (COCO);
   at 24 B, 48-D × 4 bit beats 64-D × 3 bit by 0.5 / 0.7 pp. At 32 B the two families are within 0.5 pp and the order
   differs between datasets. In this data, dropping dimensions costs less than going below about 4 bits per dimension.
4. PCA keeps 94.2 % of the variance with 32 dimensions and 99.3 % with 48; 48-D float matches 64-D float.

## Limits
- "Best per family" is a maximum over configurations on validation data (18 per budget for scalar, 8 per projection and m
  for reduce + quantize), so sub-0.5 pp differences should not be read as rankings.
- Most best reduce + quantize rows use `recon_cos`, i.e. matching on decoded m-D floats. No speed was measured here.
- Projections add a 64 × m matrix multiply per descriptor at encoding time (not timed).
- XFeat only; HPatches has been used for earlier choices (validation, not test).

## Files
`fit_projections.py`, `eval_dimred.py`, `summarize.py`; `results/projections.npz`, `results/fit_info.json`, `results/fit.log`,
`results/per_pair_*.csv.gz`, `results/summary.csv`, `results/by_type.csv`, `results/equal_bytes_all.csv`, `results/equal_bytes_best.csv`.

Run (≈ 7 min on 2 vCPU):
```
python fit_projections.py ../../data/2026-10-14_regen_cache/coco/calibration.npy results 48
python eval_dimred.py ../../data/2026-10-14_regen_cache ../2026-11-08_fair_speed/inputs/orig_code results coco
python eval_dimred.py ../../data/2026-10-14_regen_cache ../2026-11-08_fair_speed/inputs/orig_code results hpatches 0 200
python eval_dimred.py ../../data/2026-10-14_regen_cache ../2026-11-08_fair_speed/inputs/orig_code results hpatches 200 400
python eval_dimred.py ../../data/2026-10-14_regen_cache ../2026-11-08_fair_speed/inputs/orig_code results hpatches 400 580
python summarize.py results ../2026-11-08_accuracy_matrix/results/summary.csv
```
