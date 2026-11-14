"""2026-11-14 SIFT descriptors for the same calibration images and validation pairs as the XFeat experiments.

Same images, same resize (`features.resize_decode`, max side 640), same homographies as the XFeat caches:
- calibration: the 192 COCO calibration images of the 2026-10-14 manifest (unwarped);
- COCO pairs: the 64 validation images x 3 warps, using the homography stored in the XFeat pair cache. The photometric
  gain / bias of the warped image is not stored in that cache, so it is redrawn from the same ranges
  (gain U(0.82, 1.12), bias U(-12, 18)) with seed 20261114: geometry identical, brightness change not identical;
- HPatches: all 116 scenes, image 1 versus 2..6, homography S2 @ H @ inv(S1) (checked against the stored one).
SIFT: OpenCV `SIFT_create(nfeatures=1000)`, default parameters, grey image. Raw descriptors are stored as uint8.

usage: python extract_sift.py ORIG_CODE_DIR XFEAT_CACHE_DIR VAL2017.zip HPATCHES.zip OUT_DIR coco|hpatches
"""
import sys, json, time, zipfile
from pathlib import Path
import numpy as np, cv2
ORIG, XC, CZ, HZ, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5]); WHICH = sys.argv[6]
sys.path.insert(0, ORIG); import features as f
OUT.mkdir(parents=True, exist_ok=True)
sift = cv2.SIFT_create(nfeatures=1000)


def extract(img):
    kp, d = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    if d is None: return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    kp, d = kp[:1000], d[:1000]
    assert np.array_equal(d, np.round(d)) and d.max() <= 255
    return np.array([k.pt for k in kp], np.float32), d.astype(np.uint8)


def save(path, pairs):
    payload = {'count': np.array(len(pairs))}
    for i, p in enumerate(pairs):
        for k, v in p.items(): payload[f'{i}_{k}'] = np.array(v)
    np.savez(path, **payload)


t0 = time.time()
if WHICH == 'coco':
    man = json.loads((XC / 'coco' / 'manifest.json').read_text()); xp = f.read_pair_cache(XC / 'coco' / 'test_pairs.npz')
    rng = np.random.default_rng(20261114)
    with zipfile.ZipFile(CZ) as z:
        base = {Path(n).name: n for n in z.namelist() if n.lower().endswith('.jpg')}
        cal = [extract(f.resize_decode(z.read(base[n])))[1] for n in man['calibration_files']]
        np.save(OUT / 'calibration_u8.npy', np.concatenate(cal)); print('calibration descriptors', sum(map(len, cal)), round(time.time() - t0), 's', flush=True)
        pairs = []; cur = (None, None, None)
        for p in xp:
            name = p['name'].split('/')[0]
            if cur[0] != name:
                img = f.resize_decode(z.read(base[name])); cur = (name, img, extract(img))
            img = cur[1]; h, w = img.shape[:2]; k1, d1 = cur[2]
            im = cv2.warpPerspective(img, p['H'], (w, h), borderMode=cv2.BORDER_CONSTANT)
            im = np.clip(im.astype(np.float32) * rng.uniform(.82, 1.12) + rng.uniform(-12, 18), 0, 255).astype(np.uint8)
            k2, d2 = extract(im); pairs.append(dict(name=p['name'], d1=d1, d2=d2, k1=k1, k2=k2, H=p['H']))
    save(OUT / 'coco_pairs.npz', pairs); print('coco pairs', len(pairs), 'min kp', min(min(len(p['d1']), len(p['d2'])) for p in pairs), round(time.time() - t0), 's')
else:
    xp = f.read_pair_cache(XC / 'hpatches_cache' / 'hpatches_pairs.npz'); pairs = []; cache = {}; maxdiff = 0.0
    with zipfile.ZipFile(HZ) as z:
        root = 'hpatches-sequences-release/'
        def load(scene, i):
            if (scene, i) not in cache:
                raw = z.read(f'{root}{scene}/{i}.ppm'); full = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                img = f.resize_decode(raw); S = np.diag([img.shape[1] / full.shape[1], img.shape[0] / full.shape[0], 1.0])
                cache[(scene, i)] = (img, S, extract(img))
            return cache[(scene, i)]
        for p in xp:
            scene, ij = p['name'].split('/'); i, j = ij.split('-')
            if cache and next(iter(cache))[0] != scene: cache.clear()
            (_, S1, (k1, d1)), (_, S2, (k2, d2)) = load(scene, i), load(scene, j)
            Hg = np.loadtxt(z.read(f'{root}{scene}/H_{i}_{j}').decode().split()).reshape(3, 3)
            H = S2 @ Hg @ np.linalg.inv(S1); maxdiff = max(maxdiff, float(np.abs(H / H[2, 2] - p['H'] / p['H'][2, 2]).max()))
            pairs.append(dict(name=p['name'], d1=d1, d2=d2, k1=k1, k2=k2, H=p['H']))
    save(OUT / 'hpatches_pairs.npz', pairs)
    print('hpatches pairs', len(pairs), 'min kp', min(min(len(p['d1']), len(p['d2'])) for p in pairs), 'max |H - stored H|', maxdiff, round(time.time() - t0), 's')
