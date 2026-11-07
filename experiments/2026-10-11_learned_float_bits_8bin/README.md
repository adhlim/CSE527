# Learned float-bit selection + ordered 8-bin encoding

Open `2026-10-11_XFeat_learned_bits_ABS_vs_thermometer.ipynb` in Colab and Run all.
It contains both Python and C++ sources. Mount Google Drive when Colab asks.
It reuses `/content/drive/MyDrive/Colab Notebooks/CSE527_datasets/val2017.zip`.
The experiment folder is `/content/drive/MyDrive/Colab Notebooks/2026-10-11_learned_float_bits_8bin`.

## Completed training (CPU pilot, 2026-10-11)
- COCO val2017: deterministic 24-image subset, source-image-disjoint 16/3/5 train/validation/test split.
- Three known synthetic homography + brightness transformations per image; max image side 640.
- Frozen official XFeat, up to 1000 keypoints, 64-dimensional normalized float32 descriptors.
- XFeat commit: e92685f57f8318b18725c5c8c0bd28c7fe188d9a.
- 14,341 training triplets. Positives: mutual geometric correspondences with forward and inverse error <3 px.
- Hard negatives: cosine-similar descriptors whose target coordinates are >10 px from projected anchors.
- Discrete greedy selection of eight common bit positions (LSB=0): [31,25,24,23,22,26,21,20].
- A separate 256-entry ordered-bin LUT for each of 64 descriptor dimensions; shared by A and B.
- Training-only quantile bins initialize and regularize LUTs. This is not an experiment wholly independent of quantiles.
- Two coordinate-descent sweeps tune LUT values on up to 8,000 training triplets.
- Training completed in 7.39 s; this is bit search + LUT tuning only, excluding feature extraction.
- HPatches was NOT run in this pilot. Set RUN_HPATCHES=True in Colab for external evaluation only.

## Test results (5 original images, 15 transformed pairs)
Precision is aggregate correct/mutual; correct is reprojection error <3 px at resized resolution.
- Official XFeat cosine MNN: 6159/8567 = 71.8921%.
- Fixed training quantiles + integer ABS: 5698/8240 = 69.1505%.
- Learned bits + integer ABS: 5587/8168 = 68.4011%.
- Learned bits + thermometer popcount: same 5587/8168 = 68.4011%, identical MNN indices.

Median of per-pair median CPU times (3 warmups, 9 repeats):
- Official XFeat cosine MNN: 2.507 ms.
- Learned bins, uint8 working representation, encode + ABS MNN: 1.884 ms.
- Learned bins, packed 3-bit storage, encode + pack + unpack + ABS MNN: 2.157 ms.
- Learned thermometer, encode + pack + popcount MNN: 4.131 ms.
No overall XFeat speedup is claimed. XFeat extraction is excluded from matching times.
These basic C++ kernels are not the previous AVX2 SIMD4 implementation.
C++ has one matching thread; official torch cosine uses two CPU threads.
These are small synthetic-warp results, not a broad real-image benchmark.

## Storage
- Original descriptor: 256 bytes.
- Selected eight-bit patterns: intermediate only; 64 bytes if materialized.
- Bin indices: 24 bytes when packed at 3 bits/element; the fast ABS matcher uses a 64-byte unpacked workspace.
- Thermometer: 7 bits/element, 7 uint64 words = 56 bytes.
- Inference LUT: 64*256 = 16,384 bytes shared by all descriptors.
The checkpoint also contains training counts and quantile teacher thresholds; they are not per-descriptor payload.
Unseen patterns map to the corresponding dimension's rounded training mean bin.

## Files
- experiment.py / bit_codec.cpp: training, encoding, matching and verification.
- run/learned_codec.npz: fixed bit positions and learned LUTs, counts and teacher edges.
- run/training.json / selection_history.json: training history.
- run/data_manifest.json: exact source image identities and split; cache fingerprint.
- run/accuracy.csv / timing.csv / results.json: complete validation and test measurements.
- run/*_pairs.npz: actual extracted descriptor/coordinate/homography caches, no pickled objects.
- run/training_triplets.npz: positive and hard-negative descriptor triplets.
- training_run.log: original CPU execution log.
- coco_sample/: 24 original COCO sample images (included in the downloadable bundle).

## Verification
`python experiment.py --self-test` checks C++/NumPy bit-extraction equality,
3-bit pack/unpack roundtrip, all 8x8 bin pairs, full ABS/popcount distance
matrices and mutual-neighbor equality with deterministic first-index ties.
Training additionally checks the learned C++ encoder and every evaluation pair's MNN equality.

Local rerun (install numpy, scipy, torch, opencv-python-headless, tqdm; needs g++ and git):
`python experiment.py --image-dir coco_sample --count 24 --warps 3`
Use a new experiment root if changing input images or extraction settings.

The notebook first runs invariant checks, then training and held-out evaluation, then displays tables.
The displayed precomputed pilot section is explicitly labeled as the real CPU run; notebook code cells
are provided for rerunning rather than falsely marked as already executed in Colab.
