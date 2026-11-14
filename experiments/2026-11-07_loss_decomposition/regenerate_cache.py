"""Regenerate the 2026-10-14 descriptor caches with the ORIGINAL, unmodified code.

Only change: features.read_coco is replaced by a function that reproduces the exact
same image selection (same sorted 5,000-name list, same SEED) but reads the bytes
from a 256-image subset ZIP whose entries are byte-identical to val2017.zip.
"""
import sys, json, hashlib, zipfile, time
from pathlib import Path
import numpy as np

ORIG = Path(sys.argv[1])          # folder with original code (features.py, evaluate.py, ...)
DATA = Path(sys.argv[2])          # folder with coco256.zip, coco_names.json, hpatches zip
OUT = Path(sys.argv[3])           # output root (xfeat_repo must already be here)
sys.path.insert(0, str(ORIG))
import features as f
import evaluate as ev

all_names = json.loads((DATA / 'coco_names.json').read_text())['all_names']

def read_coco_subset(zip_path, count=24):
    with zipfile.ZipFile(zip_path) as z:
        names = all_names
        chosen = np.random.default_rng(f.SEED).choice(len(names), count, False)
        return [(Path(names[i]).name, f.resize_decode(z.read(names[i]))) for i in chosen]

f.read_coco = read_coco_subset

model, commit = f.load_xfeat(OUT)
t = time.time()
cal, coco, man = ev.prepare_coco(DATA / 'coco256.zip', model, OUT / 'coco', 256, 192)
print('COCO done', len(cal), len(coco), time.time() - t, flush=True)
from hpatches_source import hpatches_all
hp = hpatches_all(DATA / 'hpatches-sequences-release.zip', model, OUT / 'hpatches_cache')
print('HPatches done', len(hp), time.time() - t, flush=True)
