# 2026-11-14 — Learned thresholds (differentiable thermometer / relaxed XOR)

Purpose: answer the advisor's question "XOR is not differentiable; can you come up with a differentiable approximation?"
and test whether thresholds trained with a matching loss beat calibrated (quantile) thresholds at 2–3 bits per dimension,
where calibration matters most.

## Data
- Training: 600 synthetic pairs from 200 COCO val2017 images; development: 192 pairs from 64 images
  (`make_train_pairs.py`, run on Colab CPU with the original `features.py`; same warp distribution as the COCO validation pairs).
  Disjoint from the 192 calibration and 64 validation images (0 overlapping file names; sha256 of both files verified).
- Ground truth (`prep_pairs.py`): mutual nearest keypoints under the homography, closer than 3 px (≈ 438 positives per pair).
- Selection (checkpoint, temperature, learning rate) uses the development pairs only.
  Validation (COCO 192 pairs, HPatches 580 pairs) is evaluated once per run with the exact evaluator; MegaDepth is untouched.

## Method (`train.py`)
- Parameters: per-dimension thresholds, 64 × (L − 1) numbers, initialised from the quantile edges (the baseline).
- Thermometer bit k of value x: hard h = [x > t_k]; soft s = sigmoid((x − t_k)/τ). Forward uses h, backward uses the
  gradient of s (straight-through estimator), so the training distance equals the test-time distance.
- `l1`: D = Σ p(1 − q) + (1 − p)q over bits. This is the relaxed XOR; for hard bits it equals the Hamming distance.
- `l2`: D = squared L2 between bin indices (index = sum of thermometer bits).
- Loss: symmetric cross-entropy over −D · scale on ground-truth correspondences (scale is a learned scalar used in training only).
- Adam, lr 2e-3 for thresholds (0.02 for the scale), one pair per step, 3,000 steps (5 epochs), thresholds re-sorted after each step.
- Sanity run: the same training started from per-dimension min–max (`uniform`) edges, a deliberately poor start.

## Results (precision @3px, %)
Development pairs (hard codes, the distance used in training):
| levels | variant | start | start value | best during training |
|---|---|---|---|---|
| 4 | l2 | quantile | 65.84 | 65.98 (τ 0.03), 65.99 (τ 0.01), 65.99 (τ 0.03, seed 1) |
| 4 | l2 | uniform | 59.10 | 65.96 |
| 4 | l1 (relaxed XOR) | quantile | 64.81 | 64.96 |
| 8 | l2 | quantile | 68.10 | 68.28 |
| 8 | l2 | uniform | 65.38 | 68.24 |
τ = 0.1 diverged (56.6 after training); its best checkpoint is the initial quantile edges.

Validation (learned = best-on-dev checkpoint; `validation_summary.csv` has every run and distance):
| levels | thresholds | HPatches idx_L1 | HPatches idx_L2 | COCO idx_L1 | COCO idx_L2 |
|---|---|---|---|---|---|
| 4 | quantile | 56.20 | 57.59 | 65.67 | 66.84 |
| 4 | learned l2 (τ 0.03) | 56.25 | 57.51 | 65.81 | 66.86 |
| 4 | learned l2 from uniform | 56.19 | 57.53 | 65.91 | 66.89 |
| 4 | learned l1 (relaxed XOR) | 56.16 | 57.34 | 65.94 | 66.82 |
| 8 | quantile | 58.51 | 59.46 | 68.64 | 69.34 |
| 8 | learned l2 | 58.46 | 59.56 | 68.44 | 69.28 |
| 8 | learned l2 from uniform | 58.45 | 59.47 | 68.42 | 69.33 |
For reference, uniform edges without training: HPatches idx_L2 52.42 (4 levels), 57.72 (8 levels).

## Observations
1. The relaxation works as an optimiser: from min–max edges it recovers 5.1 pp (4 levels) and 1.8 pp (8 levels) on HPatches
   and reaches the level of quantile edges.
2. It does not beat quantile edges: learned − quantile is within ±0.25 pp on every validation column, with both signs,
   and +0.15 to +0.18 pp on the development pairs it was selected on.
3. For per-dimension thresholds on XFeat, equal-frequency (quantile) edges are already close to the optimum of this
   matching loss; the remaining loss at 2–3 bits is a limit of independent per-dimension quantization, not of threshold placement.

## Limits
- One loss, one optimiser setting, 1–2 seeds; a different objective could behave differently.
- Only thresholds were learned. A learned transform before quantization (rotation / projection) was not tried here.
- Training and development data are synthetic COCO warps; HPatches is validation (already used for earlier choices), not test.

## Files
`make_train_pairs.py`, `2026-11-14_make_train_pairs.ipynb` (Colab), `prep_pairs.py`, `train.py`, `eval_learned.py`;
`inputs/` (verbatim `features.py`, the 2026-10-14 COCO manifest);
`results/edges_*.npz` (init / best / last), `results/log_*.json`, `results/validation_summary.csv`, `results/validation_per_pair.csv.gz`.
Training data: `CSE527/data/2026-11-14_train_pairs/` (not in git). The empty `train_pairs_2026…/` folders left in this
directory by the Drive download can be deleted.

Run: `python prep_pairs.py inputs/orig_code ../../data/2026-11-14_train_pairs`, then
`python train.py ../../data/2026-10-14_regen_cache/coco/calibration.npy ../../data/2026-11-14_train_pairs inputs/orig_code results 4 l2 0.03 0.002 3000`
(≈ 45 s per run on 2 vCPU with JAX 0.6.2), then `python eval_learned.py ../../data/2026-10-14_regen_cache inputs/orig_code results`.
