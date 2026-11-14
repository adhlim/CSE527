"""2026-11-14 dimensionality-reduction baselines: fit PCA and GFR projections on calibration descriptors only.

GFR = Gram-Schmidt Feature Reduction, Algorithm 2 of Safavigerdini et al. (CVPRW 2026), following the
function-space Gram-Schmidt of Yaghooti et al. (IEEE Trans. IT 2025):
  d_0 = mean-centred X.  At step j: nu_j = top eigenvector of cov(d_{j-1});  z_j = nu_j^T X;
  new functions = {z_j} + {z_j * z_i, i < j}  (degree-2 multilinear polynomials);
  orthonormalise them against all previous functions (inner product = sample mean; constant function included);
  d_j = d_{j-1} - sum_f <d_{j-1}, f> f  over the new orthonormal functions.
With degree 1 only, the same loop is PCA deflation (checked below).

usage: python fit_projections.py CALIBRATION.npy OUT_DIR [m_max=32]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, json, time
from pathlib import Path
import numpy as np

CAL = np.load(sys.argv[1]).astype(np.float64); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
M = int(sys.argv[3]) if len(sys.argv) > 3 else 32
N, D = CAL.shape
mean = CAL.mean(0); X = CAL - mean


def pca(X):
    w, V = np.linalg.eigh(X.T @ X / len(X)); o = np.argsort(w)[::-1]
    return V[:, o], w[o]


def gfr(X, m, degree=2, eps=0.0):
    """Returns directions (D, m), residual variance before each step, number of functions used."""
    N = len(X); d = X.copy()
    F = 1 + m + (m * (m - 1) // 2 if degree == 2 else 0)
    Q = np.empty((F, N), np.float32); Q[0] = 1.0; k = 1          # orthonormal functions: mean(q*q) = 1
    Z = np.empty((m, N)); nus = []; resid = []; dropped = 0
    for j in range(m):
        w, V = np.linalg.eigh(d.T @ d / N); nu = V[:, -1]; lam = w[-1]
        if lam <= eps ** 2: break
        if nu[np.argmax(np.abs(nu))] < 0: nu = -nu                # fixed sign
        z = X @ nu; Z[j] = z; nus.append(nu); resid.append(float(w.sum()))
        new = [z] + ([z * Z[i] for i in range(j)] if degree == 2 else [])
        B = np.stack(new, 1)                                      # (N, j+1) float64
        for _ in range(2):                                        # block Gram-Schmidt, applied twice
            B -= Q[:k].T.astype(np.float64) @ (Q[:k].astype(np.float64) @ B / N) if k < 64 else _proj(Q, k, B, N)
        Qb, R = np.linalg.qr(B)                                   # orthonormalise inside the block
        keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
        dropped += int((~keep).sum())
        Qb = Qb[:, keep] * np.sqrt(N)                             # mean(q*q) = 1
        d -= Qb @ (Qb.T @ d / N)                                  # remove what the new functions explain
        Q[k:k + Qb.shape[1]] = Qb.T; k += Qb.shape[1]
    return np.stack(nus, 1), np.array(resid), k, dropped


def _proj(Q, k, B, N, step=128):                                  # chunked to limit memory
    out = np.zeros_like(B)
    for s in range(0, k, step):
        q = Q[s:min(k, s + step)].astype(np.float64); out += q.T @ (q @ B / N)
    return out


t0 = time.time()
Vp, wp = pca(X)
Vg1, _, _, _ = gfr(X, M, degree=1)
t1 = time.time()
Vg, resid, nfun, dropped = gfr(X, M, degree=2)
t2 = time.time()


def subspace_gap(A, B):                                           # 0 = same subspace
    return float(np.linalg.norm(A @ A.T - B @ B.T))


info = dict(
    date='2026-11-14', n_calibration=N, dim=D, m_max=M,
    gfr_degree='2, multilinear (z_i * z_j, i < j), constant function included',
    check_degree1_equals_pca={m: subspace_gap(Vg1[:, :m], Vp[:, :m]) for m in (8, 16, 24, 32, 48) if m <= M},
    gfr_orthonormality_error=float(np.linalg.norm(Vg.T @ Vg - np.eye(Vg.shape[1]))),
    gfr_vs_pca_subspace_gap={m: subspace_gap(Vg[:, :m], Vp[:, :m]) for m in (8, 16, 24, 32, 48) if m <= M},
    gfr_vs_pca_principal_angles_deg_max={m: float(np.degrees(np.arccos(np.clip(np.linalg.svd(Vg[:, :m].T @ Vp[:, :m], compute_uv=False).min(), -1, 1))))
                                         for m in (8, 16, 24, 32, 48) if m <= M},
    pca_variance_kept={m: float(wp[:m].sum() / wp.sum()) for m in (8, 16, 24, 32, 48)},
    gfr_linear_variance_kept={m: float(np.trace(Vg[:, :m].T @ (X.T @ X / N) @ Vg[:, :m]) / wp.sum()) for m in (8, 16, 24, 32, 48) if m <= M},
    gfr_residual_variance_fraction_before_step={int(j): float(resid[j] / wp.sum()) for j in (0, 8, 16, 24, 31) if j < len(resid)},
    pca_residual_variance_fraction_before_step={int(j): float(wp[j:].sum() / wp.sum()) for j in (0, 8, 16, 24, 31)},
    gfr_functions_used=int(nfun), gfr_functions_dropped_as_dependent=int(dropped),
    seconds=dict(pca_and_degree1=round(t1 - t0, 1), gfr_degree2=round(t2 - t1, 1)))
np.savez(OUT / 'projections.npz', mean=mean.astype(np.float32), pca=Vp.astype(np.float32), gfr=Vg.astype(np.float32),
         pca_eigenvalues=wp)
(OUT / 'fit_info.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
