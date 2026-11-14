# 2026-11-14 — Generalization to SIFT

Purpose: answer "How would it work with other detectors? SIFT?" by repeating the XFeat accuracy matrix and the
PCA + quantize comparison on SIFT (128-D), with the same images, homographies, calibration images and evaluation.
Validation only. SuperPoint is not covered (needs PyTorch; not available locally).

## Fixed
- Images and geometry of the XFeat experiments: 192 COCO calibration images, 192 COCO synthetic pairs (stored
  homographies; photometric gain / bias redrawn from the same ranges with seed 20261114), 580 HPatches pairs
  (homographies identical to the XFeat cache, max difference 0.0). Resize as before (max side 640).
- OpenCV 5.0.0 `SIFT_create(nfeatures=1000)`, default parameters; descriptor L2-normalised.
  Calibration: 158,176 descriptors. Keypoints per image: about 910 on average, minimum 36 (HPatches) / 62 (COCO).
- Code rule `count(x > edge)`, exact MNN with smallest index on ties, pooled precision at 3 px.

## Changed
Calibration strategy (6) × levels (4, 8, 16, 256) × distance (idx_L2, recon_cos; idx_L1 for 4 and 8 levels);
PCA to 64 / 32 dimensions + quantile edges (4, 16, 256 levels); native uint8 SIFT with L2 (128 B).

## Descriptor statistics (why SIFT is a different case)
SIFT values are non-negative and skewed: 17 % of calibration entries are exactly zero (up to 33 % in one dimension),
median 0.02, 99th percentile 0.285 after normalisation. XFeat values are roughly symmetric around zero.
Quantile edges collapse onto repeated values: 193 empty bins at 16 levels (none for XFeat).

## Results — HPatches (precision @3px %, with mutual / correct counts where it matters)
Float 128-D cosine (512 B): 61.07 (223,959 mutual, 136,782 correct). Native uint8 L2 (128 B): 61.08.

idx_L1:
| strategy | 4 levels (32 B) | 8 levels (48 B) |
|---|---|---|
| uniform | 62.77 | 65.75 |
| percentile | 65.86 | 67.12 |
| quantile | 60.32 | 60.93 |
| lloyd | 67.55 | 64.92 |
| uniform_shared | 61.36 | 65.19 |
| quantile_shared | 61.32 | 61.90 |

idx_L2:
| strategy | 4 | 8 | 16 | 256 |
|---|---|---|---|---|
| uniform | 59.72 | 61.01 | 61.17 | 61.37 |
| percentile | 61.86 | 62.10 | 62.21 | 62.21 |
| quantile | 60.73 | 61.35 | 61.67 | 61.87 |
| lloyd | 63.65 | 62.37 | 62.28 | 61.88 |
| uniform_shared | 59.41 | 60.76 | 60.90 | 61.06 |
| quantile_shared | 61.39 | 61.97 | 62.22 | 62.33 |

recon_cos: 58.65–60.95 at 4 levels, 60.15–61.18 at 8, 60.90–61.28 at 16, 61.05–61.10 at 256.

PCA: 64-D float (256 B) 60.05; 32-D float (128 B) 57.78; 64-D × 4 bit (32 B) 61.41 (idx_L2) / 59.44 (recon_cos);
64-D × 2 bit (16 B) 59.47; 32-D × 4 bit (16 B) 57.07; 32-D × 8 bit (32 B) 57.75.

Counts behind the highest precisions:
| configuration | mutual | correct | precision |
|---|---|---|---|
| float cosine | 223,959 | 136,782 | 61.07 |
| lloyd, 4 levels, idx_L1 | 199,146 | 134,531 | 67.55 |
| percentile, 8 levels, idx_L1 | 198,219 | 133,039 | 67.12 |
| lloyd, 4 levels, idx_L2 | 213,667 | 136,008 | 63.65 |
| quantile, 4 levels, idx_L1 | 235,782 | 142,232 | 60.32 |
Reference on every 4th HPatches pair, float descriptors: L2 62.27, RootSIFT 62.65, L1 on the L2-normalised descriptor 66.22.

COCO synthetic shows the same ordering (float 90.54; lloyd 4 levels idx_L1 92.83; quantile 4 levels idx_L1 90.53;
PCA 64-D × 4 bit idx_L2 91.02); full tables in `results/summary_sift.csv`.

## Observations
What agrees with XFeat:
1. 8 bits per dimension cost nothing (native uint8 61.08 vs float 61.07).
2. The calibration strategy matters most at low bit-widths: the spread between strategies is 7.2 pp at 4 levels (idx_L1)
   and at most 1.3 pp at 256 levels.
3. A 16× smaller code (32 B vs 512 B) keeps precision at or above the float level.

What does not agree with XFeat:
4. Best calibration. On XFeat quantile / Lloyd–Max were best and uniform worst at 2 bits. On SIFT, quantile is the worst
   per-dimension strategy in precision at 2 bits (60.3) and Lloyd–Max / percentile the best (67.6 / 65.9).
5. Distance. On XFeat, L2 and reconstruction cosine beat L1. On SIFT the order is L1 > L2 > reconstruction cosine.
   Float L1 also beats float L2 (66.2 vs 62.3 on the subset), so this is a property of SIFT and the distance, not of quantization.
6. Dimensions vs bits at 32 B. On XFeat, PCA + 4 bit beat 2 bit at the same bytes. On SIFT, 128-D × 2 bit (63.65 idx_L2,
   67.55 idx_L1) beats PCA 64-D × 4 bit (61.41).

A caution on the metric:
7. Precision can rise because fewer mutual matches survive. The highest-precision SIFT codes have 11 % fewer mutual matches
   and 1.6–2.7 % fewer correct matches than float; quantile at 4 levels has the lowest precision of the per-dimension
   strategies but the largest number of correct matches (142,232, 4 % more than float).
   Precision alone does not rank these configurations; the number of correct matches has to be reported next to it.

## Limits
- One SIFT implementation and parameter set; plain SIFT only (RootSIFT only as a float reference on a subset).
- COCO photometric change is redrawn, so COCO pairs are not pixel-identical to the XFeat ones.
- Maxima over configurations on validation data; HPatches is validation, not test. No speed measurement.

## Files
`extract_sift.py`, `sift_matrix.py`, `summarize.py`; `results/fit_sift.npz`, `results/fit_info_sift.csv`,
`results/per_pair_sift_*.csv.gz`, `results/summary_sift.csv`, `results/by_type_sift.csv`.
SIFT cache: `CSE527/data/2026-11-14_sift_cache/` (not in git).
Run (≈ 12 min): `python extract_sift.py ORIG XFEAT_CACHE val2017.zip hpatches.zip SIFT_CACHE coco`, `... hpatches`,
`python sift_matrix.py SIFT_CACHE ORIG results sift fit`, `... sift coco`, `... sift hpatches 0 200`, `... 200 400`, `... 400 580`,
`python summarize.py results sift`.
