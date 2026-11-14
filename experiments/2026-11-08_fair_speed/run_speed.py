"""2026-11-08 fair matching-speed comparison (8 bins). Diagnostic: no setting is selected here.

FIXED  : regenerated 2026-10-14 descriptor cache, ORIGINAL saved 8-bin quantile edges, exact MNN,
         2 threads everywhere, 3 warmups + 9 repetitions per pair, rotating method order.
CHANGED: matching representation / distance / implementation (see METHODS).

usage: python run_speed.py CACHE_DIR EDGES_NPZ ORIG_CODE_DIR OUT_DIR [max_pairs_per_dataset]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys, json, time, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd

CACHE, EDGES, ORIG, OUT = map(Path, sys.argv[1:5])
MAX_PAIRS = int(sys.argv[5]) if len(sys.argv) > 5 else None
sys.path.insert(0, str(ORIG)); sys.path.insert(0, str(Path(__file__).parent))
import features as f                      # original, unmodified
from optimized import ThermometerVariant  # original, unmodified
import speed_kernels as sk
import torch
torch.set_num_threads(2)

THREADS, WARMUP, REPEATS, BINS, VERIFY_PAIRS = 2, 3, 9, 8, 20
OUT.mkdir(parents=True, exist_ok=True)
edges = np.ascontiguousarray(np.load(EDGES)[f'bins_{BINS}'], np.float32)   # (64, 7)
et = np.ascontiguousarray(edges.T)                                            # (7, 64)
calibration = np.load(CACHE / 'coco' / 'calibration.npy')
INT8_SCALE = float(np.abs(calibration).max()) / 127.0   # per-tensor symmetric min-max, calibration only
INV = np.float32(1.0 / INT8_SCALE)
K = sk.Kernels(THREADS)
thermo = ThermometerVariant(BINS, edges)


def official_match(ta, tb):  # verbatim body of XFeat.match (commit e92685f), min_cossim=-1 path
    with torch.inference_mode():
        cossim = ta @ tb.t(); cossim_t = tb @ ta.t()
        _, match12 = cossim.max(dim=1); _, match21 = cossim_t.max(dim=1)
        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0
        return idx0[mutual], match12[mutual]


def blas_single_gemm(a, b):
    S = a @ b.T; r = S.argmax(1); c = S.argmax(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


# name -> (descriptor bytes, working bytes, encode(a,b)->state, match(state)->(i,j))
METHODS = {
    'official_torch_cosine': (256, 256, lambda a, b: (torch.from_numpy(a), torch.from_numpy(b)),
                              lambda s: tuple(np.asarray(x) for x in official_match(*s))),
    'float_blas_single_gemm': (256, 256, lambda a, b: (a, b), lambda s: blas_single_gemm(*s)),
    'float_cpp_fused': (256, 256, lambda a, b: (a, b), lambda s: K.mnn('float', *s)),
    'thermo8_L1_original': (56, 56, lambda a, b: (thermo.encode(a), thermo.soa(thermo.encode(b))),
                            lambda s: thermo.match(*s)),
    'byte8_L1': (64, 64, lambda a, b: (K.encode_bins(a, et, BINS), K.encode_bins(b, et, BINS)),
                 lambda s: K.mnn('l1', *s)),
    'byte8_L2': (64, 64, lambda a, b: (K.encode_bins(a, et, BINS), K.encode_bins(b, et, BINS)),
                 lambda s: K.mnn('l2', *s)),
    'int8_dot': (64, 64, lambda a, b: (K.encode_int8(a, INV), K.encode_int8(b, INV)),
                 lambda s: K.mnn('dot8', *s)),
}


def median_ms(fn):
    for _ in range(WARMUP): fn()
    v = []
    for _ in range(REPEATS):
        s = time.perf_counter(); fn(); v.append(1000 * (time.perf_counter() - s))
    return float(np.median(v))


def verify_random():
    rng = np.random.default_rng(20261108)
    for n, m in [(0, 3), (3, 0), (1, 1), (13, 19), (40, 7)]:
        for tied in (False, True):
            a = rng.normal(size=(n, 64)).astype(np.float32); b = rng.normal(size=(m, 64)).astype(np.float32)
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            if tied: a[:] = 0; b[:] = 0
            check_exact(a, b)
    for x in (et, np.nextafter(et, -np.inf), np.nextafter(et, np.inf)):   # values exactly on the edges
        assert np.array_equal(K.encode_bins(x, et, BINS), sk.ref_codes(x, edges))


def check_exact(a, b):
    ca, cb = K.encode_bins(a, et, BINS), K.encode_bins(b, et, BINS)
    assert np.array_equal(ca, sk.ref_codes(a, edges)) and np.array_equal(cb, sk.ref_codes(b, edges))
    qa, qb = K.encode_int8(a, INV), K.encode_int8(b, INV)
    assert np.array_equal(qa, sk.ref_int8(a, INV)) and np.array_equal(qb, sk.ref_int8(b, INV))
    if not len(a) or not len(b): return
    for kind, x, y in (('l1', ca, cb), ('l2', ca, cb), ('dot8', qa, qb)):
        got = K.mnn(kind, x, y); ref = sk.ref_mnn_min(sk.ref_dist(kind, x, y))
        assert all(np.array_equal(g, r) for g, r in zip(got, ref)), kind


verify_random(); print('random/edge-case verification passed', flush=True)
rows, verify_log = [], []
datasets = [('COCO_synthetic', CACHE / 'coco' / 'test_pairs.npz'),
            ('HPatches', CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')]
for label, path in datasets:
    pairs = f.read_pair_cache(path)[:MAX_PAIRS]
    names = list(METHODS)
    for ix, p in enumerate(pairs):
        a = np.ascontiguousarray(p['d1'], np.float32); b = np.ascontiguousarray(p['d2'], np.float32)
        if ix < VERIFY_PAIRS:
            check_exact(a, b); verify_log.append(p['name'])
        shift = ix % len(names); order = names[shift:] + names[:shift]
        res = {}
        for name in order:
            dbytes, wbytes, enc, match = METHODS[name]
            state = enc(a, b); i, j = match(state); res[name] = (np.asarray(i), np.asarray(j))
            enc_ms = median_ms(lambda: enc(a, b))
            match_ms = median_ms(lambda: match(state))
            total_ms = median_ms(lambda: match(enc(a, b)))
            err = np.linalg.norm(f.project(p['k1'][res[name][0]], p['H']) - p['k2'][res[name][1]], axis=1)
            scene = p['name'].split('/')[0]
            cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'
            rows.append(dict(dataset=label, category=cat, pair=p['name'], method=name, n=len(a), m=len(b),
                             mutual=len(i), correct_1px=int((err < 1).sum()), correct_3px=int((err < 3).sum()),
                             correct_5px=int((err < 5).sum()), encode_ms=enc_ms, preencoded_matching_ms=match_ms,
                             total_ms=total_ms, descriptor_bytes=dbytes, working_bytes=wbytes))
        # Same codes + same L1 + same tie rule => identical MNN (cross-check vs the original kernel).
        for x, y in zip(res['thermo8_L1_original'], res['byte8_L1']):
            assert np.array_equal(x, y), 'byte8_L1 differs from original thermometer'
        ref = set(zip(res['official_torch_cosine'][0].tolist(), res['official_torch_cosine'][1].tolist()))
        for name in ('float_blas_single_gemm', 'float_cpp_fused'):
            got = set(zip(res[name][0].tolist(), res[name][1].tolist()))
            rows[len(rows) - len(order) + order.index(name)]['same_as_official'] = len(got & ref)
        if ix % 50 == 0: print(label, ix + 1, len(pairs), flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / 'per_pair.csv', index=False)


def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    for k in ('encode_ms', 'preencoded_matching_ms', 'total_ms'):
        r[k + '_median'] = float(g[k].median()); r[k + '_mean'] = float(g[k].mean())
    r['descriptor_bytes'] = int(g.descriptor_bytes.iloc[0]); r['working_bytes'] = int(g.working_bytes.iloc[0])
    if 'same_as_official' in g and g.same_as_official.notna().any():
        r['same_as_official'] = int(g.same_as_official.sum())
    return r


order = list(METHODS)
summary = pd.DataFrame([dict(dataset=d, method=m, **summ(df[(df.dataset == d) & (df.method == m)]))
                        for d in df.dataset.unique() for m in order])
for d in summary.dataset.unique():
    base = summary[(summary.dataset == d) & (summary.method == 'float_cpp_fused')].total_ms_median.iloc[0]
    off = summary[(summary.dataset == d) & (summary.method == 'official_torch_cosine')].total_ms_median.iloc[0]
    summary.loc[summary.dataset == d, 'speedup_vs_float_cpp_fused'] = base / summary.total_ms_median
    summary.loc[summary.dataset == d, 'speedup_vs_official_torch'] = off / summary.total_ms_median
summary.to_csv(OUT / 'summary.csv', index=False)
pd.DataFrame([dict(dataset=d, category=c, method=m, **summ(g)) for (d, c, m), g in
              df.groupby(['dataset', 'category', 'method'])]).to_csv(OUT / 'by_type.csv', index=False)


def cpu_model():
    try:
        return next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        return platform.processor()


env = dict(date='2026-11-08', cpu=cpu_model(), platform=platform.platform(), numpy=np.__version__,
           torch=torch.__version__, threads=THREADS, warmup=WARMUP, repeats=REPEATS, bins=BINS,
           int8_scale=INT8_SCALE, int8_scheme='per-tensor symmetric, scale=max|x| over calibration /127, round half to even',
           kernel_source_sha256=K.source_sha256, max_pairs=MAX_PAIRS,
           exact_verification=dict(random_cases=True, real_pairs=verify_log,
                                   thermo8_vs_byte8_L1_identical_all_pairs=True),
           scope='CPU encode + exact MNN; XFeat extraction excluded; reported time = median over pairs of per-pair median')
(OUT / 'environment.json').write_text(json.dumps(env, indent=2))
pd.set_option('display.width', 250)
print(summary[['dataset', 'method', 'precision_3px', 'correct_3px', 'encode_ms_median', 'preencoded_matching_ms_median',
               'total_ms_median', 'speedup_vs_float_cpp_fused', 'speedup_vs_official_torch', 'descriptor_bytes']].round(3).to_string(index=False))
