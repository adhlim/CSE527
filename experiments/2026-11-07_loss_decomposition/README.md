# 2026-11-07 — Loss decomposition: quantization vs distance function

Purpose: split the accuracy gap between float cosine and the 8-bin quantile thermometer
into (i) information lost by quantization and (ii) loss caused by the distance function
(bin-index L1 = thermometer XOR/popcount). Diagnostic only: no setting is selected here.

## Fixed
- Frozen XFeat (commit e92685f…, weights sha256 0f5187fd…), max side 640, top-k 1000.
- Descriptor/keypoint cache regenerated with the unmodified 2026-10-14 code (see below).
- Quantization edges: the ORIGINAL saved `fixed_quantile_edges.npz` (8 and 16 bins), not refitted.
- Exact MNN, smallest index on ties (same rule as the C++ kernels), pooled precision at 1/3/5 px.

## Changed (only the space/function used for the MNN distance)
| ID | Representation used for distance | Distance |
|---|---|---|
| A | float descriptor | cosine (reference) |
| B | bin → per-dimension calibration conditional mean, renormalized | cosine |
| C | same reconstruction, not renormalized | squared L2 |
| D | integer bin index | L1 (= current thermometer method) |
| E | integer bin index | squared L2 |
| D_tiebreak_float | D, ties broken by float cosine | diagnostic only, not deployable |

Reconstruction values use calibration descriptors only (`recon_table_*bin.npy`).

## Cache regeneration check
`regenerate_cache.py` imports the original code unchanged; only `read_coco` is redirected to a
256-image subset ZIP whose entries are byte-identical to `val2017.zip` (same 5,000-name list and seed).
- Calibration/test file lists and descriptor count (191,581) identical to the 2026-10-14 manifest.
- Descriptor bytes are NOT bitwise identical (sha256 c8b4b051… vs 237ababf…; recomputed edges differ by ≤2.1e-7),
  attributed to a different CPU/BLAS. Therefore the original saved edges are used.
- Float and 8-bin results vs the 2026-10-14 per-pair CSVs: COCO 192/192 pairs identical;
  HPatches 578/580 pairs identical (float correct_3px 176,695 vs 176,696; 8-bin mutual 286,966 vs 286,965).

## Data status
HPatches was already used to choose methods in earlier experiments; it is reported here as a
validation set, not as an independent test set. COCO internal pairs are disjoint from calibration
by source image.

## Files
`decomposition.py`, `regenerate_cache.py`, `check_equiv.py`, `results/summary.csv`, `results/by_type.csv`,
`results/per_pair.csv`, `results/ties_summary.csv`, `results/ties_per_pair.csv`, `results/recon_table_*bin.npy`.
`same_as_float` = number of mutual matches identical to the float-cosine MNN set.
