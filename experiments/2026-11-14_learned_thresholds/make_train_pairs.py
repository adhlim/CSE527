"""2026-11-14 training / development pairs for threshold learning (run on Colab; needs torch + opencv).

Creates synthetic-homography pairs with the ORIGINAL, unmodified `features.make_pairs` (same warp distribution as the
COCO validation pairs) from COCO val2017 images that were NOT used for calibration (192) or validation (64).
  train: 200 images x 3 warps = 600 pairs      dev: 64 images x 3 warps = 192 pairs (for early stopping / choices)
HPatches, the COCO validation pairs and MegaDepth are not touched.

usage: python make_train_pairs.py ORIG_CODE_DIR COCO_MANIFEST.json VAL2017.zip WORK_DIR OUT_DIR
"""
import sys, json, hashlib, zipfile, time
from pathlib import Path
import numpy as np

ORIG, MANIFEST, ZIP, WORK, OUT = map(Path, sys.argv[1:6])
sys.path.insert(0, str(ORIG))
import features as f

SEED_SELECT, SEED_TRAIN, SEED_DEV = 20261114, 20261115, 20261116
N_TRAIN, N_DEV, WARPS = 200, 64, 3
OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)

man = json.loads(MANIFEST.read_text())
used = set(man['calibration_files']) | set(man['test_files'])
assert len(used) == 256, len(used)
with zipfile.ZipFile(ZIP) as z:
    names = sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
    assert len(names) == 5000, len(names)
    base = {Path(n).name: n for n in names}
    assert used <= set(base), 'manifest images missing from the zip'
    free = sorted(set(base) - used)                                   # 4,744 unused images
    pick = np.random.default_rng(SEED_SELECT).choice(len(free), N_TRAIN + N_DEV, False)
    chosen = [free[i] for i in pick]
    train_names, dev_names = chosen[:N_TRAIN], chosen[N_TRAIN:]
    assert not (set(chosen) & used) and len(set(chosen)) == N_TRAIN + N_DEV
    load = lambda ns: [(n, f.resize_decode(z.read(base[n]))) for n in ns]
    train_imgs, dev_imgs = load(train_names), load(dev_names)

import torch
model, commit = f.load_xfeat(WORK)
t0 = time.time()
train = f.make_pairs(train_imgs, model, WARPS, SEED_TRAIN)
dev = f.make_pairs(dev_imgs, model, WARPS, SEED_DEV)
f.save_pair_cache(OUT / 'train_pairs.npz', train); f.save_pair_cache(OUT / 'dev_pairs.npz', dev)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(OUT / 'train_manifest.json').write_text(json.dumps(dict(
    date='2026-11-14', source='COCO val2017', xfeat_commit=commit, device=str(model.dev), torch=torch.__version__,
    top_k=1000, max_side=640, warps_per_image=WARPS, seeds=dict(select=SEED_SELECT, train=SEED_TRAIN, dev=SEED_DEV),
    train_files=train_names, dev_files=dev_names, train_pairs=len(train), dev_pairs=len(dev),
    excluded='192 calibration + 64 validation images of the 2026-10-14 manifest',
    extraction_seconds=time.time() - t0,
    sha256=dict(train=sha(OUT / 'train_pairs.npz'), dev=sha(OUT / 'dev_pairs.npz'))), indent=2))
print('train pairs', len(train), 'dev pairs', len(dev), 'seconds', round(time.time() - t0))
