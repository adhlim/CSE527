# 2026-11-08 — Fair matching-speed comparison (8 bins)

Purpose: measure the matching-speed gain against a float baseline implemented at the same
optimization level, and measure the speed of bin-index L2 and an NN-quantization-style int8
baseline. The 2026-10-14 speedup (3.75×) used XFeat's `match()`, which computes two GEMMs in
PyTorch, so it mixed representation gains with baseline overhead and implementation differences.
Diagnostic only: no setting is selected from these results.

## Fixed
- Regenerated 2026-10-14 descriptor cache (`CSE527/data/2026-10-14_regen_cache`), max side 640, top-k 1000.
- Original 8-bin quantile edges (`inputs/fixed_quantile_edges.npz`, copied unchanged).
- Exact MNN, smallest index on ties; 2 threads; 3 warmups + 9 repeats per pair; method order rotates per pair.
- Timing excludes XFeat extraction. Reported time = median over pairs of the per-pair median.
  Means are also saved. `encode`, `preencoded_matching`, and `total` (= encode + match) are separate columns.

## Changed
| method | stored B/desc | description |
|---|---:|---|
| official_torch_cosine | 256 | verbatim body of XFeat `match` (two GEMMs), min_cossim=-1 |
| float_blas_single_gemm | 256 | NumPy/BLAS one GEMM + row/column argmax |
| float_cpp_fused | 256 | AVX2 FMA, fused MNN loop shared with the integer arms |
| thermo8_L1_original | 56 | unchanged 2026-10-14 thermometer encoder + SIMD4 kernel |
| byte8_L1 | 64 | 1 byte/dim bin index, SAD (same MNN as thermometer, checked on every pair) |
| byte8_L2 | 64 | 1 byte/dim bin index, squared L2 (maddubs) |
| int8_dot | 64 | per-tensor symmetric int8, scale = max abs(calibration)/127, integer dot |

All new C++ arms share one fused MNN loop that processes 4 candidates per step (like the
original SIMD4 kernel). AVX2 + FMA only (no AVX-512/VNNI).

## Verification built into `run_speed.py`
- Random and edge-value cases: encoders and integer MNN equal NumPy references exactly.
- First 20 pairs of each dataset: integer arms equal NumPy references exactly.
- Every pair: `byte8_L1` MNN identical to the original thermometer kernel.
- Float arms: overlap with the official matcher is stored as `same_as_official`
  (float summation order differs, so exact equality is not required).

## Data status
HPatches is a validation set (already used for earlier method choices). Planned untouched final test:
MegaDepth-1500 (pose AUC), not used yet.

## Files
- `run_speed.py`, `speed_kernels.py`: experiment code. `inputs/orig_code/`: verbatim copies of
  2026-10-14 `features.py`, `optimized.py` (sha256 75846c92…, f368cb92…).
- `2026-11-08_fair_speed.ipynb`: Colab runner (official run).
- `results_cloud_preliminary_20pairs/`: 20 pairs per dataset on a shared cloud VM
  (Intel Xeon 2.10 GHz, 2 vCPU). Used only to check that the code runs; NOT the reported timing.
