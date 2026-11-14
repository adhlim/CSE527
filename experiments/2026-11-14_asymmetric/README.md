# 2026-11-14 — Asymmetric matching (one side float, one side compressed)

Purpose: the PQ experiment measured an asymmetric setting only for PQ. This measures it for 64-D scalar quantization and
for PCA reduce + scalar quantization, so that the three families are compared in the same setting.
The asymmetric setting corresponds to a compressed database queried with uncompressed descriptors (e.g. localization
against a stored map). Validation only; no setting is selected here.

## Fixed
- Regenerated 2026-10-14 XFeat cache; exact MNN, smallest index on ties; pooled precision at 1/3/5 px.
- 64-D scalar family: the edges saved by `2026-11-08_accuracy_matrix` (6 strategies × 5 level counts).
- PCA family: projection saved by `2026-11-14_dim_reduction`, m = 48/32/24/16, 4/8/16/256 levels, percentile or quantile edges.
- Reconstruction value of a bin = calibration conditional mean.

## Changed
- `symmetric`: cosine between two reconstructed descriptors (= `recon_cos`).
- `asymmetric`: cosine between the float descriptor of image 1 and the reconstructed descriptor of image 2
  (in the reduced space for PCA).

## Check
Symmetric 64-D rows equal `recon_cos` of the accuracy matrix for all 60 configurations (max difference 0.001 pp).

## Results (precision @3px, %, best configuration per cell; float: HPatches 60.59, COCO 70.57)
HPatches:
| bytes | scalar sym | scalar asym | PCA+scalar sym | PCA+scalar asym | PQ sym | PQ asym |
|---|---|---|---|---|---|---|
| 8 | — | — | 53.08 | 56.43 | 52.85 | 57.17 |
| 16 | 57.38 | 59.02 | 59.10 | 59.51 | 59.31 | 60.00 |
| 24 | 59.55 | 60.13 | 59.97 | 60.27 | — | — |
| 32 | 60.39 | 60.52 | 59.90 | 59.91 | 60.45 | 60.52 |

COCO synthetic:
| bytes | scalar sym | scalar asym | PCA+scalar sym | PCA+scalar asym | PQ sym | PQ asym |
|---|---|---|---|---|---|---|
| 8 | — | — | 62.98 | 66.46 | 61.28 | 66.62 |
| 16 | 66.51 | 68.40 | 69.54 | 70.00 | 68.94 | 69.92 |
| 24 | 69.13 | 69.94 | 70.05 | 70.38 | — | — |
| 32 | 70.29 | 70.38 | 70.52 | 70.51 | 70.46 | 70.52 |

(Scalar symmetric here is `recon_cos` only; with `idx_L2` the 16 B value is 57.59 / 66.84.)

## Observations
1. Keeping one side in float recovers a large part of the loss at small budgets: +1.6 pp (scalar, 16 B),
   +3.4 pp (PCA + scalar, 8 B), +4.3 pp (PQ, 8 B) on HPatches. At 32 B the gain is ≤ 0.15 pp.
2. In the asymmetric setting the three families are within 1 pp of each other at 16 B (59.0 / 59.5 / 60.0 on HPatches)
   and within 0.1 pp at 32 B (except PCA 32-D, limited by the projection itself at 59.9).
3. At 16 B asymmetric, the loss against float is 0.6–1.6 pp; at 24 B it is ≤ 0.5 pp for both measured families.

## Limits
Maxima over configurations on validation data; no speed measurement; XFeat only; HPatches is validation, not test.

## Files
`asym.py`; `results/per_pair_*.csv.gz`, `results/summary.csv`, `results/by_type.csv`, `results/equal_bytes_sym_vs_asym.csv`.
Run: `python asym.py CACHE ORIG ../2026-11-08_accuracy_matrix/results/edges_all.npz ../2026-11-14_dim_reduction/results/projections.npz results coco`,
then `... hpatches 0 200`, `... hpatches 200 400`, `... hpatches 400 580` (≈ 6 min on 2 vCPU). Summary tables were built
from the per-pair files with the pandas snippet recorded in this folder's `summarize.py`.
