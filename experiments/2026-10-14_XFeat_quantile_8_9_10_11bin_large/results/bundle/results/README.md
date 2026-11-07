# XFeat quantile thermometer comparison — 2026-10-14

This experiment compares the frozen official XFeat cosine matcher with the existing fixed per-dimension quantile thermometer implementation at 8, 9, 10, and 11 bins. It does not train float-bit selection, a pattern-to-bin LUT, a projection, or matching-optimized bin boundaries.

Calibration: 192 source images randomly selected from COCO val2017; 191,581 normalized 64D descriptors. Thresholds are fitted once separately for each descriptor dimension and bin count.

Internal evaluation: 64 disjoint COCO source images × 3 known geometric/photometric transforms = 192 pairs. External evaluation: all 116 HPatches sequences, reference against targets 2–6 = 580 pairs (57 illumination and 59 viewpoint sequences). Total: 772 evaluated pairs.

Evaluation uses the same descriptor/keypoint cache for all five matchers. Images have maximum side 640 and at most 1,000 keypoints. Resized HPatches homographies are corrected using S2 H inv(S1). The original XFeat matcher uses min_cossim=-1; no extra score filtering is imposed.

Precision is pooled correct/mutual at <1, <3, and <5 pixel homography reprojection error. This is a full-image keypoint matching test and is not the official HPatches patch benchmark or MMA metric. Correct counts and counts per pair are included to distinguish precision from the number of correct matches.

Timing is CPU-only, with two Torch/OpenMP threads, 3 warmups and 9 repetitions on every evaluated pair. Total time includes direct bin encoding, thermometer packing, B layout conversion and exact MNN, but excludes XFeat feature extraction. Benchmark method order rotates across pairs. Reported time is the median of per-pair medians. Scalar/SIMD MNN equivalence is checked on every evaluated pair.

Storage per descriptor: float32 256 B, 8-bin 56 B, 9-bin 64 B, 10-bin 72 B, 11-bin 80 B. Coordinates/scores, temporary matching buffers, and shared thresholds are not included in those descriptor-only sizes. The shared thresholds and fitting times are supplied separately.

The self-contained Colab notebook uses the two existing ZIP archives in MyDrive/Colab Notebooks/CSE527_datasets. It regenerates the data and saves new result CSVs, manifests, and fixed edges. The XFeat Git commit is pinned in features.py. The archive includes full per-pair results and dataset filename splits; the large raw archives/descriptors are excluded because they can be reproduced from those settings.


## Ordinary binary notation extension
The same notebook now adds ordinary packed 3-bit/8-bin and 4-bit/16-bin indices. A 16-bin thermometer control is included to isolate notation from quantization. Within each bin count, both representations share exactly the same threshold arrays and ordinal bin assignments. No PCA, k-means, float-bit extraction, or learned LUT is introduced. Existing 8/9/10/11-bin thresholds are checked for bitwise equality with the previously saved thresholds.

Ordinary binary is compared using integer absolute differences, not XOR Hamming distance. The 3-bit implementation packs each 64D descriptor into 24 bytes and expands it once per matching call to a 64-byte working representation; expansion time is included in both preencoded and total matching time. The 4-bit implementation uses 32 bytes and computes packed nibble ABS directly. A 16-bin thermometer uses 15 bitplanes (120 bytes). On every evaluation pair, the two encodings at 8 bins and at 16 bins must return identical MNN indices.

The extension reruns all eight methods on the same 192 synthetic and 580 HPatches pairs, with the same calibration split and same features. Remeasured times belong to this run; they should not be mixed with earlier timing measurements.
