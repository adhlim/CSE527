"""2026-11-14 MegaDepth-1500 held-out evaluation of the 15 pre-registered configurations (run on Colab).

Protocol: pair list, intrinsics and ground-truth poses from XFeat's assets/megadepth_1500.json (the LoFTR split);
pose estimation and AUC as in XFeat's evaluation script (see md_pose.py). Nothing is fitted on MegaDepth.
Extraction settings (both registered; predictions are judged on A):
  A  project setting: features.resize_decode (longest side 640, multiples of 32) + features.extract (top-k 1000),
     i.e. the setting of the calibration descriptors.
  B  benchmark setting of the XFeat script: image resized to the size stored in the json (longest side 1200),
     detectAndCompute with top-k 4096. Thresholds calibrated in setting A are applied unchanged.
Keypoints are mapped back to original pixel coordinates, where the intrinsics of the json apply.
Results are written per (setting, configuration); an existing complete file is skipped, so the run can be resumed.

usage: python md_eval.py DATA_ROOT JSON ORIG_CODE_DIR XFEAT_WORK_DIR FROZEN.npz OUT_DIR SETTINGS CONFIGS PREREG_COMMIT
       SETTINGS e.g. A,B ; CONFIGS e.g. C0 or all ; PREREG_COMMIT = git commit hash of the pre-registration (or 'check')
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, cv2

DATA, JSON, ORIG, WORK, FROZEN, OUT = [Path(x) for x in sys.argv[1:7]]
SETTINGS = sys.argv[7].split(','); CONFIGS = sys.argv[8]; COMMIT = sys.argv[9]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ORIG))
import md_configs as mc, md_pose as mp, features as f
SEEDS = [0, 1, 2, 3, 4]
OUT.mkdir(parents=True, exist_ok=True)
cfgs = list(mc.CONFIGS) if CONFIGS == 'all' else CONFIGS.split(',')
pairs = json.loads(JSON.read_text()); root = DATA / 'megadepth_test_1500'
assert len(pairs) == 1500 and root.exists(), (len(pairs), root)
P = mc.load(FROZEN); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
todo = [(s, c) for s in SETTINGS for c in cfgs if not ((OUT / f'pairs_{s}_{c}.csv').exists() and len(pd.read_csv(OUT / f'pairs_{s}_{c}.csv')) == 1500)]
print('to run:', todo, flush=True)

import torch
torch.set_num_threads(2)
model = None


def extract_all(setting):
    global model
    if model is None: model, _ = f.load_xfeat(WORK)
    feats = {}; t0 = time.time()
    sizes = {}
    for e in pairs:
        for n, hw, sc in ((e['pair_names'][0], e['size0_hw'], e['scale0']), (e['pair_names'][1], e['size1_hw'], e['scale1'])): sizes[n] = (hw, sc)
    for ix, (name, (hw, sc)) in enumerate(sorted(sizes.items())):
        raw = (root / name).read_bytes(); full = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR); H0, W0 = full.shape[:2]
        assert abs(hw[1] * sc[0] - W0) < 1.5 and abs(hw[0] * sc[1] - H0) < 1.5, (name, hw, sc, full.shape)   # json scale maps to the original size
        if setting == 'A':
            img = f.resize_decode(raw); k, d = f.extract(model, img); scale = np.array([W0 / img.shape[1], H0 / img.shape[0]], np.float32)
        else:
            img = cv2.resize(full, (hw[1], hw[0])); t = torch.tensor(img.astype(np.float32) / 255).permute(2, 0, 1)[None].to(model.dev)
            with torch.inference_mode(): o = model.detectAndCompute(t, top_k=4096)[0]
            k = o['keypoints'].cpu().numpy().astype(np.float32); d = o['descriptors'].cpu().numpy().astype(np.float32)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12); scale = np.array(sc, np.float32)
        feats[name] = (k * scale[None], np.ascontiguousarray(d))
        if ix % 100 == 0: print(f'extract {setting} {ix + 1}/{len(sizes)} kpts {len(k)} {time.time() - t0:.0f}s', flush=True)
    return feats


for setting in SETTINGS:
    run = [c for (s, c) in todo if s == setting]
    if not run: continue
    feats = extract_all(setting)
    nk = np.array([len(v[0]) for v in feats.values()]); print(f'setting {setting}: {len(feats)} images, keypoints mean {nk.mean():.0f} min {nk.min()}', flush=True)
    for c in run:
        t0 = time.time(); rows = []
        for e in pairs:
            (k0, d0), (k1, d1) = feats[e['pair_names'][0]], feats[e['pair_names'][1]]
            i, j = mc.match(c, P, d0, d1); p0, p1 = k0[i], k1[j]
            K0, K1, T = np.array(e['K0'], np.float64), np.array(e['K1'], np.float64), np.array(e['T_0to1'], np.float64)
            r = dict(pair_id=e['pair_id'], scene=e['scene_id'], matches=len(i))
            for s in SEEDS:
                err, inl = mp.pose_error(p0, p1, K0, K1, T, s * 100003 + int(e['pair_id'])); r[f'err_s{s}'] = err
                if s == 0: r['inliers_s0'] = inl
            rows.append(r)
            if len(rows) % 300 == 0: print(f'{setting} {c} {len(rows)}/1500 {time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(OUT / f'pairs_{setting}_{c}.csv', index=False)
    del feats

# ---------------- summary over everything present ----------------
summ = []
for setting in ('A', 'B'):
    for c, (desc, nbytes) in mc.CONFIGS.items():
        p = OUT / f'pairs_{setting}_{c}.csv'
        if not p.exists(): continue
        d = pd.read_csv(p)
        if len(d) != 1500: continue
        a = pd.DataFrame([mp.error_auc(d[f'err_s{s}'].to_numpy()) for s in SEEDS]) * 100
        row = dict(setting=setting, config=c, description=desc, bytes=nbytes, matches_mean=d.matches.mean(),
                   inlier_ratio_s0=float((d.inliers_s0 / d.matches.clip(lower=1)).mean()))
        for k in a.columns: row[f'{k}_mean'] = a[k].mean(); row[f'{k}_std'] = a[k].std(ddof=1)
        summ.append(row)
S = pd.DataFrame(summ); S.to_csv(OUT / 'summary.csv', index=False)
import poselib
(OUT / 'manifest.json').write_text(json.dumps(dict(
    date=time.strftime('%Y-%m-%d %H:%M:%S'), preregistration_commit=COMMIT, settings=SETTINGS, configs=cfgs, seeds=SEEDS,
    frozen_configs_sha256=sha(FROZEN), pairs_json_sha256=sha(JSON), xfeat_commit=f.XFEAT_COMMIT,
    code_sha256={n: sha(HERE / n) for n in ('md_eval.py', 'md_configs.py', 'md_pose.py')}, features_py_sha256=sha(Path(ORIG) / 'features.py'),
    ransac=dict(max_epipolar_error=mp.RANSAC_THR, success_prob=mp.CONF, min_iterations=20, max_iterations=1000),
    versions=dict(numpy=np.__version__, torch=torch.__version__, poselib=poselib.__version__, opencv=cv2.__version__),
    device=str(model.dev) if model is not None else 'not loaded in this call'), indent=1))
pd.set_option('display.width', 220)
print(S[['setting', 'config', 'bytes', 'matches_mean', 'auc@5_mean', 'auc@10_mean', 'auc@10_std', 'auc@20_mean']].round(2).to_string(index=False) if len(S) else 'no complete results yet')
