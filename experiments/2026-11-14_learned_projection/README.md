# 2026-11-14 — Learned projection + thresholds (the simplest learned compression)

Purpose: after per-dimension thresholds alone showed no gain from learning (`2026-11-14_learned_thresholds`), test whether
learning the linear transform before quantization helps. This is the minimal form of what learned compression
(e.g. Differentiable Product Quantization) does, and it tests the advisor's first question: is the method data-independent?

## Fixed
- Frozen XFeat descriptors; 600 training / 192 development synthetic COCO pairs of `2026-11-14_learned_thresholds`
  (disjoint from calibration and validation images). Checkpoint selection on development pairs only.
- Start: W = top-m PCA directions, thresholds = quantile edges of the projected calibration descriptors.
- Relaxation and loss as in `2026-11-14_learned_thresholds` (straight-through thermometer, bin-index L2, symmetric
  cross-entropy); sigmoid temperature = 0.25 × per-dimension std; Adam, 3,000 steps, lr 1e-3 for W unless noted.
- Validation with exact MNN on COCO synthetic (192 pairs) and HPatches (580 pairs).

## Changed
(m, levels): (16,16) and (32,4) = 8 B; (32,16) and (64,4) = 16 B; (48,16) = 24 B; (64,16) = 32 B.
`both` = W and thresholds trained; `edges` = thresholds only. One extra run with lr 1e-4 and 600 steps ("mild").

## Results (precision @3px, %; float 64-D: COCO 70.57, HPatches 60.59, development 69.36)
"Training-free best" = best configuration at that byte budget from `2026-11-08_accuracy_matrix`,
`2026-11-14_dim_reduction` and `2026-11-14_product_quantization` (symmetric).
Learned rows: better of idx_L2 / recon_cos on each dataset.
| bytes | configuration | development | COCO synthetic | HPatches |
|---|---|---|---|---|
| 8 | training-free best | — | 64.47 | 53.80 |
| 8 | learned 16-D × 4 bit | 67.77 | 68.94 | 53.25 |
| 8 | learned 32-D × 2 bit | 66.51 | 67.70 | 53.51 |
| 16 | training-free best | — | 69.61 | 59.31 |
| 16 | learned 32-D × 4 bit, thresholds only | 68.47 | 69.72 | 59.02 |
| 16 | learned 32-D × 4 bit, mild (lr 1e-4, 600 steps) | 68.62 | 69.69 | 58.83 |
| 16 | learned 32-D × 4 bit | 69.20 | 70.44 | 58.16 |
| 16 | learned 64-D × 2 bit | 68.10 | 69.29 | 55.84 |
| 24 | training-free best | — | 70.05 | 60.04 |
| 24 | learned 48-D × 4 bit | 69.32 | 70.50 | 58.61 |
| 32 | training-free best | — | 70.53 | 60.45 |
| 32 | learned 64-D × 4 bit | 69.34 | 70.53 | 58.13 |

HPatches by type at 16 B (32-D × 4 bit): viewpoint idx_L2 45.02 (start) → 46.81 (learned), but the training-free
`recon_cos` of the same start already gives 48.76; illumination 66.90 (start, recon_cos) vs 66.78 (learned).

## Observations
1. On data from the training distribution (synthetic COCO warps) learning helps, most at small budgets:
   +4.5 pp at 8 B, +0.8 pp at 16 B, and it reaches float accuracy at 16–32 B on the development pairs.
2. On HPatches the learned models are at or below the training-free best at every budget:
   −0.3 to −0.5 pp at 8 B, −1.2 pp at 16 B, −1.4 pp at 24 B, −2.3 pp at 32 B.
3. The more the projection is trained, the larger the gap between the two datasets: thresholds only (69.72 / 59.02),
   mild (69.69 / 58.83), full (70.44 / 58.16) at 16 B. The learned transform adapts to the synthetic-warp distribution.
4. Calibration-only compression (quantile edges, PCA) transfers across the two datasets; the learned transform does not,
   with this training set. A learned method needs training data that covers the target conditions, which is the
   reason DPQ is trained per scene.

## Limits
- Training data are synthetic homography warps of COCO images with mild photometric change; HPatches contains real
  illumination and viewpoint changes. A more varied training set could change observation 2. Not tested.
- One loss, one seed, linear map only; no regularisation of W (e.g. orthogonality) was tried.
- The "mild" run was added after seeing the HPatches result of the full run, so it is a diagnostic, not a selected setting.
- HPatches is a validation set (used for earlier choices); MegaDepth-1500 is untouched.

## Files
`train_proj.py`, `eval_proj.py`; `results/model_*.npz` (mean, W_init, W_best, edges), `results/log_*.json`,
`results/validation_summary.csv`, `results/validation_by_type.csv`, `results/validation_per_pair.csv.gz`.
Run: `python train_proj.py ../../data/2026-10-14_regen_cache/coco/calibration.npy ../../data/2026-11-14_train_pairs ../2026-11-14_learned_thresholds/inputs/orig_code results 32 16 both 0.001 3000`
(≈ 40 s per run), then `python eval_proj.py ../../data/2026-10-14_regen_cache ../2026-11-14_learned_thresholds/inputs/orig_code results`.
