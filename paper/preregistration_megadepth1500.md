# Pre-registration: final test on MegaDepth-1500

**Status: READY TO FREEZE (2026-11-14); float-only pipeline check passed.** All protocol items are settled (Section 7). The registered run (Step 2 of the
notebook) must not start before (1) the float-only pipeline check of Section 6 has passed and (2) this file and
`experiments/2026-11-14_megadepth1500/` are committed to git. The commit hash is passed to the run and stored in its manifest.

## 1. Why this file exists
Every accuracy number so far comes from COCO synthetic pairs and HPatches, and these two sets were used to compare several
hundred configurations. MegaDepth-1500 has not been used for anything in this project. It is run once, with the
configurations, metric and predictions fixed below, and every result is reported whether or not it agrees with validation.

## 2. Rules
- The configuration list in Section 4 is run in full, once. Nothing is added, removed or re-tuned after results are seen.
- A re-run is allowed only for failures unrelated to the numbers (crash, data-loading or pose-code bug found by the
  float-only checks of Section 6). The list stays unchanged and the reason is recorded.
- Any configuration added after the results are seen is reported separately as post-hoc.
- All thresholds, projections, codebooks and learned parameters are the frozen files of Section 3. Nothing is fitted on MegaDepth.

## 3. Frozen artifacts (sha256, first 16 hex digits)
| File | sha256 |
|---|---|
| `data/2026-10-14_regen_cache/coco/calibration.npy` (191,581 COCO descriptors) | 6d5c99f522e270c9 |
| `experiments/2026-11-08_accuracy_matrix/results/edges_all.npz` | b712795352951dd6 |
| `experiments/2026-11-14_dim_reduction/results/projections.npz` (mean, pca, gfr) | ed159d33bdaa04e0 |
| `experiments/2026-11-14_product_quantization/results/codebooks_pca64.npz` | c4bc59c55e0f3efd |
| `experiments/2026-11-14_learned_thresholds/results/edges_L4_l2_tau0.03_lr0.002_s0.npz` (key `best`) | 913012995685d1ac |
| `experiments/2026-11-14_learned_projection/results/model_m32_L16_both_lrw0.001_s0.npz` (`W_best`, `edges_best`) | f5eac780804a4ea8 |
| `features.py` (original extraction code) | 75846c92e96eeafe |

All parameters derived from the files above are collected in one file that the run reads and nothing else:

| File | sha256 |
|---|---|
| `experiments/2026-11-14_megadepth1500/frozen_configs.npz` | 8666ae9e8db35dab |
| `experiments/2026-11-14_megadepth1500/inputs/megadepth_1500.json` (pair list, intrinsics, poses; from the XFeat repository) | 6aa01108aa26b1d5 |

`build_frozen.py` builds `frozen_configs.npz` and checks that `md_configs.match` reproduces the validation precision of
all 15 configurations on both validation sets (30 of 30 values equal to two decimals, 2026-11-14).

XFeat: commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a, frozen weights.
Reconstruction tables (`recon_cos`) = calibration conditional mean per bin, recomputed from `calibration.npy` with the frozen edges.
Edges for PCA / GFR configurations = per-dimension quantile edges of the projected calibration descriptors (rule of `eval_dimred.py`).
int8: per-tensor symmetric, scale = max|calibration| / 127 = 0.0046016.

## 4. Configurations (15)
Code rule `count(x > edge)`; mutual nearest neighbour, smallest index on ties; both images compressed (symmetric).
| ID | Configuration | Bytes | Distance |
|---|---|---|---|
| C0 | float32, 64-D | 256 | cosine |
| C1 | int8 per-tensor, 64-D | 64 | integer dot |
| C2 | 64-D, 16 levels, percentile edges | 32 | bin-index L2 |
| C3 | 64-D, 8 levels, quantile edges (= original thermometer method) | 24 (56 as thermometer) | bin-index L1 |
| C4 | same codes as C3 | 24 | bin-index L2 |
| C5 | same codes as C3 | 24 | reconstruction cosine |
| C6 | 64-D, 4 levels, quantile edges | 16 | bin-index L2 |
| C7 | 64-D, 4 levels, uniform (min–max) edges | 16 | bin-index L2 |
| C8 | 64-D, 4 levels, learned thresholds | 16 | bin-index L2 |
| C9 | PCA 32-D, 16 levels, quantile edges | 16 | reconstruction cosine |
| C10 | GFR 32-D, 16 levels, quantile edges | 16 | reconstruction cosine |
| C11 | PCA 48-D, 16 levels, quantile edges | 24 | reconstruction cosine |
| C12 | PQ, PCA rotation, M = 16, K = 256 | 16 | reconstructed squared L2 |
| C13 | learned projection 32-D + learned thresholds, 16 levels | 16 | bin-index L2 |
| C14 | PCA 32-D, float | 128 | cosine |

## 5. Metric and predictions
Primary metric: pose AUC at 10° (AUC@5° and AUC@20°, number of matches and inlier ratio are also reported).
RANSAC is repeated with 5 fixed seeds; mean and standard deviation over seeds are reported. "Δ" below is a difference of
seed-means in AUC@10 points. A prediction counts as confirmed only if it holds for the mean.

| # | Prediction | Feedback item it answers |
|---|---|---|
| P1 | C1 is within 0.5 of C0 | exact gain: 4× memory at no accuracy cost |
| P2 | C2 is within 1.5 of C0 | exact gain: 8× memory |
| P3 | C6 − C7 ≥ 2.0 (quantile beats uniform at 2 bits) | calibration strategies |
| P4 | C4 ≥ C3 and C5 ≥ C3 (L2 and cosine are not worse than L1 on the same codes) | "what if you use L2 or cosine?" |
| P5 | \|C8 − C6\| ≤ 1.0 (learned thresholds do not beat quantile) | differentiable XOR / fine-tuned thresholds |
| P6 | C9 − C6 ≥ 1.0 (at 16 B, fewer dimensions beat fewer bits) | "what if you change the quantization?" |
| P7 | \|C10 − C9\| ≤ 1.0 (GFR ≈ PCA on XFeat) | comparison with Gram-Schmidt Feature Reduction |
| P8 | \|C12 − C9\| ≤ 1.5 (training-free PQ ≈ PCA + scalar at 16 B) | comparison with the product-quantization family |
| P9 | C13 < C9 (the projection learned on synthetic COCO warps does not transfer) | dataset independence |
| P10 | C14 < C2 (128 B of reduced float is worse than 32 B of 4-bit codes) | advantages / limitations vs dimension reduction |

Validation values behind the predictions (HPatches precision@3px): C0 60.59, C1 60.58, C2 60.25, C3 58.51, C4 59.46, C5 59.55,
C6 57.59, C7 52.42, C8 57.51, C9 59.10, C10 59.11, C11 59.97, C12 59.31, C13 58.01, C14 59.91.
Precision@3px and pose AUC are different metrics, so the numeric margins above are judgement calls made before seeing any
MegaDepth number.

## 6. Pipeline check before the run (float descriptors only)
Step 1 of the notebook runs C0 only, in both settings. In setting B, which is the protocol of the XFeat evaluation
script, the result is compared with the values published for XFeat on MegaDepth-1500: AUC@5 / 10 / 20 = 42.6 / 56.4 / 67.7.
A gap of more than about 3 points at AUC@10 is treated as a pipeline problem to be fixed before Step 2.
This check uses float descriptors only, so it does not inform any choice among C1-C14.

**Check result (Colab CPU, 2026-11-14): passed.** C0 in setting B: AUC@5 / 10 / 20 = 42.73 / 56.52 / 67.75 (published 42.6 / 56.4 / 67.7),
1,468 matches per pair on average, standard deviation of AUC@10 over the 5 seeds 0.34. C0 in setting A: 25.90 / 40.78 / 55.71,
390 matches per pair, standard deviation 0.29. No configuration, margin or protocol item was changed after this check.
Already verified locally (2026-11-14): `md_pose.py` recovers synthetic two-view poses (median error 0.09 degrees with
0.5 px noise and 50 % outliers), returns identical results for a repeated seed, and gives AUC 1 / 0 for perfect / failed poses.

## 7. Protocol (settled)
- **Pairs.** The 1,500 MegaDepth pairs, intrinsics and relative poses of `inputs/megadepth_1500.json` (LoFTR split as
  distributed with XFeat). Images: the archive downloaded by XFeat's `modules/dataset/download.py`.
- **Extraction, setting A (primary; predictions are judged here).** `features.resize_decode` (longest side 640, sides
  rounded down to multiples of 32) and `features.extract` (top-k 1000): the setting of the calibration descriptors.
- **Extraction, setting B (secondary, robustness).** Image resized to the size stored in the json (longest side 1200),
  `detectAndCompute` with top-k 4096, as in XFeat's `modules/eval/megadepth1500.py`. Thresholds, projections and
  codebooks calibrated in setting A are applied without re-calibration.
- **Matching.** Mutual nearest neighbour without a similarity threshold, smallest index on ties (`md_configs.py`).
- **Pose.** PoseLib `estimate_relative_pose`, max epipolar error 2.5 px, success probability 0.99999, 20-1000
  iterations, keypoints in original pixel coordinates (`md_pose.py`, parameters of the XFeat script).
- **Seeds.** 5 per pair (PoseLib seed and a shuffle of the correspondences). Reported: mean and standard deviation of AUC over seeds.
- **Error and metric.** max(rotation error, translation-direction error); AUC at 5, 10, 20 degrees. Also reported:
  number of matches and inlier ratio.
- **Judging.** `md_predictions.py` (written before the run) evaluates P1-P10 on setting A, seed-mean AUC@10.
- **Environment.** Colab CPU runtime; device and library versions are stored in `results_colab/manifest.json`.
- Advisor review of this list: not received as of 2026-11-14.

## 8. Not covered by this run
- Other descriptors (SIFT, SuperPoint): separate validation experiment on HPatches.
- Matching speed: measured separately (`2026-11-08_fair_speed`, `2026-11-08_scaling`); no timing claim is made from this run.
- Differentiable Product Quantization and the authors' GFR code are not reproduced; C12 and C10 are our own
  training-free PQ and our re-implementation of the published GFR algorithm.
