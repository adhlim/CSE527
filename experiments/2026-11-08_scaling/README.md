# 2026-11-08 — Scaling: exact 1-NN search time vs database size

Purpose: test whether compact codes give larger speed gains when the database grows beyond the
two-image setting (1000 × 1000), e.g. matching against a map. Speed only: no accuracy, no selection.

## Fixed
- Queries: XFeat descriptors (1000 each) of 3 distinct COCO test source images (`test_pairs` 0, 3, 6, `d1`).
- Database: HPatches XFeat descriptors (reference image once per scene + every target image), first N.
- Exact 1-NN, smallest database index on ties; 2 threads; database encoding is offline and not timed.
- Every integer method uses the same cache-blocked loop (64 queries × 4096 database entries per tile,
  4 candidates per step). Float uses NumPy/BLAS GEMM per 16,384-entry block with a running row argmax.
- Edges: 8 levels = 2026-10-14 quantile edges; 16 levels = percentile edges from the accuracy matrix.
  int8 = per-tensor symmetric, scale = max|calibration| / 127.

## Changed
N ∈ {1,024; 4,096; 16,384; 65,536; 262,144; 524,288} and method:
| method | B/desc | distance |
|---|---:|---|
| float32_blas | 256 | dot (cosine on unit vectors) |
| int8_dot | 64 | integer dot |
| byte8_L1 | 64 | bin index L1 |
| byte8_L2 | 64 | bin index L2 |
| nib16_L2_packed | 32 | 4-bit packed bin index L2 |
| thermo8_L1 | 56 | thermometer XOR/popcount (database plane-major) |

## Verification (built in)
Every method's 1-NN index equals a NumPy reference exactly (70 queries × 5,003 database entries, plus N = 1, 3, 4097).

## Timing
1 warmup; 9 repeats (N ≤ 16k), 5 (N ≤ 64k), 3 (larger); median per query set, then median over the 3 query sets.
`per_query.csv` keeps every repeat.

## Files
`run_scaling.py`, `scale_kernels.py`, `inputs/` (verbatim `features.py`, edges), `2026-11-08_scaling.ipynb` (Colab runner),
`results_cloud_preliminary/`: shared cloud VM (Intel Xeon 2.10 GHz, 2 vCPU), code check only, not the reported timing.

## Runs (2026-11-07)
- `results_colab_*`: Colab CPU (Xeon 2.20 GHz, 2 vCPU). Too noisy to report (repeat max/min up to 3.2×,
  query-set spread up to 2.7×, non-monotonic in N). Kept for the record only.
- `results_laptop_run1/`, `results_laptop_run2/`: local VM on Intel Core Ultra 7 155H (2 vCPU), AC power,
  two consecutive runs (an interrupted earlier attempt of run 2 was discarded). For N ≥ 65,536 the two runs agree
  within about ±10 % per method; for N ≤ 4,096 (1–5 ms per call) they do not agree and are not interpreted.
