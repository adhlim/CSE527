# CSE527 — XFeat Descriptor Compression and Matching

Experiments on quantization, descriptor representations, learned transforms, and CPU matching for XFeat local descriptors.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/` | 12 standalone notebooks covering early experiments and unified comparisons |
| `experiments/success-direct-thermometer-AVX2-SIMD4/` | 8/9/10/11-bin notebooks, the official XFeat comparison baseline, and saved measurements |
| `experiments/2026-10-11_learned_float_bits_8bin/` | Learned bit/LUT experiment, Python and C++ sources, checkpoint, training records, and results |
| `experiments/2026-10-14_XFeat_quantile_8_9_10_11bin_large/` | Later quantile experiment and its original result bundle |
| `experiments/2026-10-14_XFeat_quantile_8_9_10_11bin_large/results/bundle/` | Unpacked bundle, including five Python source files and benchmark manifests/tables |
| `results/hpatches_5methods/` | Five-method HPatches comparison tables and environment/data manifests |
| `results/historical/` | Earlier experimental measurements, parameter files, and saved source snippets |

## Suggested starting points

1. [Quantile experiment](experiments/2026-10-14_XFeat_quantile_8_9_10_11bin_large/2026-10-14_XFeat_quantile_8_9_10_11bin_large.ipynb).
2. [Fair comparison against official XFeat](experiments/success-direct-thermometer-AVX2-SIMD4/2026-10-08_fair_comparison_8_11bin_vs_official_XFeat.ipynb).
3. [Unified five-method comparison](notebooks/2026-10-11_XFeat_5methods_unified_comparison.ipynb).
4. [Learned bit/LUT experiment documentation](experiments/2026-10-11_learned_float_bits_8bin/README.md).

Filenames such as `official_XFeat.ipynb` are the original experiment filenames; consult each notebook for its baseline implementation and settings.

## Running the experiments

Open a notebook in Google Colab, read its setup cells, install its dependencies, and mount Google Drive when requested. The migration preserves original notebook contents, outputs, and code. Paths referring to `/content/drive/MyDrive/Colab Notebooks/` still refer to the original Drive layout; they were not automatically rewritten to the GitHub layout.

The repository does not include all input data. Restore the required inputs from the original Drive folders, or update the notebook's input/output paths before running. Local AVX2 experiments require a compatible CPU/compiler. Dependency and thread settings vary between experiments; consult their setup cells and saved environment manifests.

For the learned bit/LUT experiment, consult its README and use `python experiment.py --self-test` to run the provided invariant checks after installing dependencies. This migration did not rerun model extraction, training, benchmarks, or that self-test.

## Data and provenance

The migration imported **189 original files** from Google Drive on **2026-11-07 (Asia/Seoul)**. The original quantile ZIP is also unpacked into `results/bundle/` for browsing; these are additional copies of its contents.

- `migration_manifest.json` records each imported file's Drive ID, original path, destination, byte size, and SHA-256.
- `excluded_files.json` lists files left on Drive with the reason for exclusion.
- Dataset archives (`hpatches-sequences-release.zip`, `val2017.zip`), extracted training/evaluation input caches, personal sample photographs, and generated assembly dumps remain on Drive.
- Small trained parameters and quantization thresholds (`.npz`, `.npy`, `.pt`) are retained where available.
- Files named `official_match_source.py` are saved indented method excerpts for provenance, rather than standalone executable modules.
- Existing experiment READMEs and outputs describe the original runs and may mention input files not distributed here. Saved results are historical measurements, not newly verified performance claims.

## Migration validation

Downloaded original files were checked against Drive metadata byte sizes. Notebook JSON structure and Python syntax were checked (saved indented method snippets were checked after dedenting). The migration also checked for common credential patterns. No experiment results or algorithms were changed.
