"""2026-11-08 scaling experiment: exact 1-NN search time vs database size. Speed only (no accuracy, no selection).

FIXED  : query sets = XFeat descriptors of 3 COCO test source images (1000 each, disjoint from the database source),
         database = XFeat descriptors of HPatches images (concatenated, first N), 2 threads, exact 1-NN with
         smallest index on ties, database encoded offline (not timed), cache-blocked loops for every arm.
CHANGED: database size N and representation/distance.

usage: python run_scaling.py CACHE_DIR INPUTS_DIR OUT_DIR [quick]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys, json, time, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd

CACHE, INPUTS, OUT = map(Path, sys.argv[1:4])
QUICK = len(sys.argv) > 4
sys.path.insert(0, str(INPUTS / 'orig_code')); sys.path.insert(0, str(Path(__file__).parent))
import features as f
import scale_kernels as sk

OUT.mkdir(parents=True, exist_ok=True)
SIZES = [1024, 4096, 16384] if QUICK else [1024, 4096, 16384, 65536, 262144, 524288]
K = sk.ScaleKernels(2)
q8 = np.load(INPUTS / 'fixed_quantile_edges.npz')['bins_8']            # 8-level quantile (2026-10-14)
p16 = np.load(INPUTS / 'edges_all.npz')['percentile_16']               # 16-level percentile (accuracy matrix)
cal = np.load(CACHE / 'coco' / 'calibration.npy')
INV = np.float32(127.0 / float(np.abs(cal).max()))

# ---------------- data ----------------
hp = f.read_pair_cache(CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')
seen, chunks = set(), []
for p in hp:
    scene = p['name'].split('/')[0]
    if scene not in seen: seen.add(scene); chunks.append(p['d1'])
    chunks.append(p['d2'])
DB_ALL = np.ascontiguousarray(np.concatenate(chunks), np.float32)
coco = f.read_pair_cache(CACHE / 'coco' / 'test_pairs.npz')
QUERIES = [np.ascontiguousarray(coco[k]['d1'], np.float32) for k in (0, 3, 6)]   # 3 different source images
assert DB_ALL.shape[0] >= SIZES[-1], DB_ALL.shape
print('database pool', DB_ALL.shape, 'queries', [len(q) for q in QUERIES], flush=True)


def float_nn(q, db, block=16384):  # BLAS GEMM per database block, running row argmax (first index on ties)
    best = np.full(len(q), -np.inf, np.float32); arg = np.zeros(len(q), np.int64)
    for s in range(0, len(db), block):
        S = q @ db[s:s + block].T; j = S.argmax(1); v = S[np.arange(len(q)), j]
        upd = v > best; best[upd] = v[upd]; arg[upd] = j[upd] + s
    return arg


# name -> (bytes per database descriptor, encode(x) -> representation, search(qrep, dbrep))
def thermo_db(x): return np.ascontiguousarray(sk.thermo_aos(sk.codes(x, q8), 7).T)   # SoA (planes, N)
METHODS = {
    'float32_blas': (256, lambda x: x, lambda q, d: float_nn(q, d)),
    'int8_dot': (64, lambda x: sk.int8_codes(x, INV), lambda q, d: K.nn('dot8', q, d)),
    'byte8_L1': (64, lambda x: sk.codes(x, q8), lambda q, d: K.nn('l1', q, d)),
    'byte8_L2': (64, lambda x: sk.codes(x, q8), lambda q, d: K.nn('l2', q, d)),
    'nib16_L2_packed': (32, lambda x: sk.pack4(sk.codes(x, p16)), lambda q, d: K.nn('n4', q, d)),
    'thermo8_L1': (56, None, lambda q, d: K.nn('thermo', q, d, planes=7)),
}
QENC = dict(thermo8_L1=lambda x: np.ascontiguousarray(sk.thermo_aos(sk.codes(x, q8), 7)))
DENC = dict(thermo8_L1=thermo_db)


# ---------------- verification: every kernel == NumPy reference (exact) ----------------
def verify():
    rng = np.random.default_rng(1108)
    q = QUERIES[0][:70]; db = DB_ALL[rng.choice(len(DB_ALL), 5003, replace=False)]
    c8q, c8d = sk.codes(q, q8).astype(np.int64), sk.codes(db, q8).astype(np.int64)
    c16q, c16d = sk.codes(q, p16).astype(np.int64), sk.codes(db, p16).astype(np.int64)
    iq, idb = sk.int8_codes(q, INV).astype(np.int64), sk.int8_codes(db, INV).astype(np.int64)
    L1 = np.abs(c8q[:, None] - c8d[None]).sum(2)
    ref = {'byte8_L1': L1, 'thermo8_L1': L1, 'byte8_L2': ((c8q[:, None] - c8d[None]) ** 2).sum(2),
           'nib16_L2_packed': ((c16q[:, None] - c16d[None]) ** 2).sum(2), 'int8_dot': -(iq @ idb.T)}
    for name, (_, enc, search) in METHODS.items():
        qe = QENC.get(name, enc)(q); de = DENC.get(name, enc)(db)
        got = np.asarray(search(qe, de))
        exp = (q @ db.T).argmax(1) if name == 'float32_blas' else sk.ref_nn(ref[name])
        assert np.array_equal(got, exp), name
    # tiny edge cases: database not a multiple of 4 / of the 4096 block
    for m in (1, 3, 4097):
        d = DB_ALL[:m]
        for name in ('byte8_L1', 'nib16_L2_packed', 'thermo8_L1', 'int8_dot'):
            _, enc, search = METHODS[name]
            got = np.asarray(search(QENC.get(name, enc)(q), DENC.get(name, enc)(d)))
            if name == 'thermo8_L1' or name == 'byte8_L1':
                R = np.abs(c8q[:, None] - sk.codes(d, q8).astype(np.int64)[None]).sum(2)
            elif name == 'nib16_L2_packed':
                R = ((c16q[:, None] - sk.codes(d, p16).astype(np.int64)[None]) ** 2).sum(2)
            else:
                R = -(iq @ sk.int8_codes(d, INV).astype(np.int64).T)
            assert np.array_equal(got, sk.ref_nn(R)), (name, m)
    print('exact verification passed', flush=True)


verify()


def timed(fn, n_rep):
    fn(); v = []
    for _ in range(n_rep):
        s = time.perf_counter(); fn(); v.append(1000 * (time.perf_counter() - s))
    return float(np.median(v)), v


rows = []
for N in SIZES:
    db = DB_ALL[:N]
    reps = 9 if N <= 16384 else 5 if N <= 65536 else 3
    enc = {name: DENC.get(name, e)(db) for name, (_, e, _) in METHODS.items()}
    names = list(METHODS)
    for qi, q in enumerate(QUERIES):
        order = names[qi % len(names):] + names[:qi % len(names)]
        for name in order:
            nbytes, e, search = METHODS[name]
            qe = QENC.get(name, e)(q); de = enc[name]
            med, allv = timed(lambda: search(qe, de), reps)
            rows.append(dict(N=N, query=qi, method=name, db_bytes_per_desc=nbytes, db_megabytes=N * nbytes / 2**20,
                             ms_median=med, ms_all=json.dumps([round(x, 3) for x in allv]),
                             ns_per_distance=1e6 * med / (len(q) * N)))
    print('N', N, 'done', flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / 'per_query.csv', index=False)
s = df.groupby(['N', 'method']).agg(ms=('ms_median', 'median'), ns_per_distance=('ns_per_distance', 'median'),
                                    db_megabytes=('db_megabytes', 'first')).reset_index()
base = s[s.method == 'float32_blas'].set_index('N').ms
s['speedup_vs_float32_blas'] = s.apply(lambda r: base[r.N] / r.ms, axis=1)
s.to_csv(OUT / 'summary.csv', index=False)


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    except Exception: return ''


(OUT / 'environment.json').write_text(json.dumps(dict(
    date='2026-11-08', platform=platform.platform(), numpy=np.__version__, threads=2, sizes=SIZES,
    lscpu=sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|cache|MHz'"), kernel_source_sha256=K.source_sha256,
    queries='COCO test pairs 0,3,6 d1 (3 distinct source images, 1000 desc each)',
    database='HPatches descriptors (reference image once per scene + all targets), first N', task='exact 1-NN',
    repeats='9 (N<=16k), 5 (N<=64k), 3 (larger); 1 warmup; median per query, then median over 3 queries',
    edges=dict(byte8_and_thermo8='2026-10-14 quantile 8 levels', nib16='percentile 16 levels (accuracy matrix)'),
    int8_scale=float(1 / INV), quick=QUICK), indent=2))
pd.set_option('display.width', 200)
print(s.pivot(index='method', columns='N', values='ms').round(2).to_string())
print(s.pivot(index='method', columns='N', values='speedup_vs_float32_blas').round(2).to_string())
