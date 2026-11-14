# 2026-11-14 — Product quantization baseline (k-means PQ) at equal bytes

Purpose: give a number for the vector-quantization family that the advisor pointed to (Differentiable Product
Quantization, Laskar et al., ECCV 2024). DPQ itself (scene-specific auto-encoder, metric losses, localization pipeline)
is NOT reproduced; this is the training-free building block it starts from: k-means PQ.
Validation only (COCO synthetic 192 pairs, HPatches 580 pairs). No setting is selected for the final test here.

## Fixed
- Regenerated 2026-10-14 XFeat cache; exact MNN, smallest index on ties, pooled precision at 1/3/5 px
  (same evaluation as `2026-11-08_accuracy_matrix`).
- Codebooks: k-means (Lloyd, random init, 25 iterations, seed 20261114) on a fixed random subset of 65,536 of the
  191,581 COCO calibration descriptors (256 points per centroid at K = 256, the FAISS default).

## Changed
- (M sub-vectors, K centroids): (8,256) (16,16) = 8 B; (16,256) (32,16) (64,4) = 16 B; (32,256) (64,16) = 32 B.
- Rotation: `none` (consecutive raw dimensions) or `pca64` (full PCA rotation fitted on calibration, dimensions
  interleaved so that every sub-vector receives similar variance; a simple stand-in for OPQ).
- Distance: `sdc_cos` / `sdc_l2` (both images compressed, reconstructed, cosine / squared L2) and
  `adc_cos` (first image float, second image compressed and reconstructed).

## Checks
- (64, 16) and (64, 4) without rotation are 1-D k-means per dimension, i.e. the scalar Lloyd–Max quantizer:
  HPatches sdc_cos 60.35 and 56.73 vs 60.39 and 56.73 for `lloyd` + `recon_cos` in the accuracy matrix.
- Float row reproduces 60.59 (HPatches) and 70.57 (COCO).

## Results (precision @3px, %; float 64-D: HPatches 60.59, COCO 70.57)
Symmetric (both sides compressed), best distance of sdc_cos / sdc_l2:
| bytes | (M, K) | HPatches none | HPatches pca64 | COCO none | COCO pca64 |
|---|---|---|---|---|---|
| 8 | (8, 256) | 46.55 | 52.64 | 53.78 | 60.93 |
| 8 | (16, 16) | 47.33 | 52.85 | 54.92 | 61.28 |
| 16 | (16, 256) | 57.20 | 59.31 | 65.40 | 68.94 |
| 16 | (32, 16) | 57.04 | 58.86 | 65.36 | 68.43 |
| 32 | (32, 256) | 60.38 | 60.45 | 70.21 | 70.46 |

Three families at equal bytes (best configuration of each; HPatches / COCO):
| bytes | 64-D scalar | PCA/GFR reduce + scalar | PQ symmetric | PQ asymmetric (one side float) |
|---|---|---|---|---|
| 8 | — | 53.80 / 64.47 | 52.85 / 61.28 | 57.17 / 66.62 |
| 16 | 57.59 / 66.84 | 59.11 / 69.61 | 59.31 / 68.94 | 60.00 / 69.92 |
| 32 | 60.39 / 70.29 | 59.92 / 70.53 | 60.45 / 70.46 | 60.52 / 70.52 |

## Observations
1. At 32 B the three families are within about 0.5 pp of each other and of float; vector quantization gives no clear
   gain over 4-bit scalar quantization there.
2. At 16 B, PQ needs the PCA rotation to help: without it PQ (57.2 / 65.4) is at the level of 2-bit scalar
   quantization (57.6 / 66.8); with it (59.3 / 68.9) it is at the level of PCA 32-D × 4 bit (59.1 / 69.6).
   The gain at small budgets comes from using the correlation between dimensions, which either a rotation
   or a projection provides.
3. At 8 B every training-free method loses 7–8 pp or more in the symmetric setting.
4. When only one side is compressed (database compressed, query float, the visual-localization setting of DPQ),
   16 B loses 0.6 pp and 8 B loses 3.4–4.0 pp.

## Limits
- "Best" values are maxima over a few configurations on validation data; differences below 0.5 pp are not rankings.
- The asymmetric setting was measured only for PQ; scalar and reduce + quantize would also improve in that setting,
  so column 5 must not be compared with columns 2–3.
- No speed measurement. Symmetric PQ matching here reconstructs 64-D floats; a table-lookup implementation was not written.
- k-means was run once (one seed); XFeat only; HPatches is validation, not test.

## Files
`pq.py`, `summarize.py`; `results/codebooks_{none,pca64}.npz`, `results/fit_info_*.csv`, `results/environment.json`,
`results/per_pair_*.csv.gz`, `results/summary.csv`, `results/by_type.csv`, `results/equal_bytes_three_families.csv`.

Run (≈ 10 min on 2 vCPU): `python pq.py CACHE ORIG results fit none`, `... fit pca64`, then
`... coco 0 96`, `... coco 96 192`, `... hpatches 0 200`, `... hpatches 200 400`, `... hpatches 400 580`, then
`python summarize.py results ../2026-11-08_accuracy_matrix/results/summary.csv ../2026-11-14_dim_reduction/results/equal_bytes_best.csv`.
