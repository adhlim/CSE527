"""2026-11-14 learned thresholds: train per-dimension quantization thresholds with a matching loss.

FIXED  : frozen XFeat descriptors (64-D); training pairs = 600 synthetic COCO pairs disjoint from calibration and
         validation images; initial thresholds = quantile edges of the calibration descriptors (the baseline);
         code rule count(x > edge); selection only on the 192 development pairs.
CHANGED: thresholds (64 x (L-1) numbers) are updated by gradient descent.

Differentiable relaxation: thermometer bit k of a value x is hard h = [x > t_k]; its soft version is
s = sigmoid((x - t_k) / tau). The forward pass uses h, the backward pass uses the gradient of s (straight-through).
  variant l1 : D = sum over bits of p(1-q) + (1-p)q  -> relaxed XOR; equals the Hamming distance for hard bits
  variant l2 : D = squared L2 between bin indices (index = sum of thermometer bits)
Loss: symmetric cross-entropy over -D * scale for ground-truth correspondences (scale is a learned scalar, training only).

usage: python train.py CAL.npy PAIR_DIR ORIG_CODE_DIR OUT_DIR LEVELS VARIANT TAU LR STEPS [SEED] [INIT=quantile|uniform]
(INIT=uniform starts from per-dimension min-max edges: a deliberately poor start used to check that training works)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2')
import sys, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

CAL = np.load(sys.argv[1]).astype(np.float32); PD = Path(sys.argv[2]); ORIG = sys.argv[3]; OUT = Path(sys.argv[4])
L = int(sys.argv[5]); VAR = sys.argv[6]; TAU = float(sys.argv[7]); LR = float(sys.argv[8]); STEPS = int(sys.argv[9])
SEED = int(sys.argv[10]) if len(sys.argv) > 10 else 0
INIT = sys.argv[11] if len(sys.argv) > 11 else 'quantile'
sys.path.insert(0, ORIG); import features as f
OUT.mkdir(parents=True, exist_ok=True)
TAG = f'L{L}_{VAR}_tau{TAU}_lr{LR}_s{SEED}' + ('' if INIT == 'quantile' else f'_{INIT}')
EVAL_EVERY = 300; LR_SCALE = 0.02; NEG = 1e9

tr = np.load(PD / 'train_arrays.npz'); D1, D2, TGT, N2 = tr['d1'], tr['d2'], tr['tgt'].astype(np.int32), tr['n2'].astype(np.int32)
dev = f.read_pair_cache(PD / 'dev_pairs.npz')
edges0 = np.quantile(CAL, np.arange(1, L) / L, axis=0).T.astype(np.float32)          # (64, L-1) baseline
if INIT == 'uniform':
    lo, hi = CAL.min(0), CAL.max(0); edges0 = (lo[:, None] + (hi - lo)[:, None] * (np.arange(1, L) / L)[None]).astype(np.float32)


def bits(x, t):
    z = x[:, :, None] - t[None]
    s = jax.nn.sigmoid(z / TAU); h = (z > 0).astype(jnp.float32)
    return s + jax.lax.stop_gradient(h - s)                                         # hard forward, soft backward


def dist(a, b, t):
    p, q = bits(a, t), bits(b, t)
    if VAR == 'l1':
        p, q = p.reshape(len(a), -1), q.reshape(len(b), -1)
        return p @ (1 - q).T + (1 - p) @ q.T
    ia, ib = p.sum(-1), q.sum(-1)
    return (ia * ia).sum(1)[:, None] + (ib * ib).sum(1)[None] - 2 * ia @ ib.T


def loss_fn(params, a, b, tgt, n2):
    t, logs = params
    logits = -dist(a, b, t) * jnp.exp(logs)
    logits = jnp.where(jnp.arange(b.shape[0])[None] < n2, logits, -NEG)             # padded second-image rows
    w = (tgt >= 0).astype(jnp.float32); j = jnp.maximum(tgt, 0)
    row = jax.nn.log_softmax(logits, axis=1)[jnp.arange(a.shape[0]), j]
    col = jax.nn.log_softmax(jnp.where(w[:, None] > 0, logits, -NEG), axis=0)[jnp.arange(a.shape[0]), j]
    return -((row + col) * w).sum() / (2 * w.sum())


vg = jax.jit(jax.value_and_grad(loss_fn))


def codes(x, e):
    out = np.empty(x.shape, np.uint8)
    for d in range(64): out[:, d] = np.searchsorted(e[d], x[:, d], side='left')
    return out


def dev_precision(e):
    tot = cor = 0
    for p in dev:
        a, b = codes(p['d1'], e).astype(np.float32), codes(p['d2'], e).astype(np.float32)
        if VAR == 'l1': Dm = np.abs(a[:, None, :] - b[None, :, :]).sum(2) if False else _l1(a, b)
        else: Dm = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        r = Dm.argmin(1); c = Dm.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); j = r[i]
        err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1); tot += len(i); cor += int((err < 3).sum())
    return 100 * cor / tot


def _l1(a, b):                                                                       # exact index L1 via thermometer bits
    ta = (a[:, :, None] > np.arange(L - 1)[None, None]).reshape(len(a), -1).astype(np.float32)
    tb = (b[:, :, None] > np.arange(L - 1)[None, None]).reshape(len(b), -1).astype(np.float32)
    return ta @ (1 - tb).T + (1 - ta) @ tb.T


# initial scale: logits spread of about 10 between typical distances
d0 = np.asarray(dist(jnp.asarray(D1[0]), jnp.asarray(D2[0]), jnp.asarray(edges0)))
params = [jnp.asarray(edges0), jnp.asarray(np.log(10.0 / d0.std()), jnp.float32)]
m = [jnp.zeros_like(p) for p in params]; v = [jnp.zeros_like(p) for p in params]
rng = np.random.default_rng(SEED); order = rng.permutation(len(D1)); t0 = time.time()
log = [dict(step=0, dev_precision_3px=dev_precision(edges0), seconds=0.0)]
best = (log[0]['dev_precision_3px'], 0, edges0.copy()); run = 0.0
print(TAG, log[0], flush=True)
for step in range(1, STEPS + 1):
    k = order[(step - 1) % len(order)]
    if (step - 1) % len(order) == len(order) - 1: order = rng.permutation(len(D1))
    val, g = vg(params, jnp.asarray(D1[k]), jnp.asarray(D2[k]), jnp.asarray(TGT[k]), int(N2[k]))
    run += float(val)
    for i, lr in enumerate((LR, LR_SCALE)):                                           # Adam, two learning rates
        m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
        params[i] = params[i] - lr * (m[i] / (1 - 0.9 ** step)) / (jnp.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
    params[0] = jnp.sort(params[0], axis=1)                                           # keep thresholds ordered
    if step % EVAL_EVERY == 0 or step == STEPS:
        e = np.asarray(params[0]); pr = dev_precision(e)
        log.append(dict(step=step, train_loss=run / (EVAL_EVERY if step % EVAL_EVERY == 0 else step % EVAL_EVERY),
                        dev_precision_3px=pr, mean_abs_edge_shift=float(np.abs(e - edges0).mean()),
                        scale=float(np.exp(params[1])), seconds=round(time.time() - t0, 1)))
        run = 0.0; print(log[-1], flush=True)
        if pr > best[0]: best = (pr, step, e.copy())
np.savez(OUT / f'edges_{TAG}.npz', best=best[2], last=np.asarray(params[0]), init=edges0)
(OUT / f'log_{TAG}.json').write_text(json.dumps(dict(tag=TAG, levels=L, variant=VAR, tau=TAU, lr=LR, lr_scale=LR_SCALE, steps=STEPS,
    seed=SEED, init=INIT, best_dev=best[0], best_step=best[1], init_dev=log[0]['dev_precision_3px'], log=log, jax=jax.__version__), indent=1))
print('best dev', round(best[0], 3), 'at step', best[1], 'init', round(log[0]['dev_precision_3px'], 3))
