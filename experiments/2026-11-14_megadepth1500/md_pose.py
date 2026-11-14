"""Relative-pose estimation and AUC, following the XFeat MegaDepth-1500 evaluation script
(modules/eval/megadepth1500.py at commit e92685f, itself adapted from LoFTR): PoseLib relative pose with
max_epipolar_error = 2.5 px, success_prob = 0.99999, 20..1000 iterations; pose error = max(rotation error,
translation-direction error) in degrees; AUC at 5 / 10 / 20 degrees.
Added here: a seed. PoseLib is given `seed`, and the correspondences are also shuffled with the same seed, so that
repeated runs sample different minimal sets.
"""
import numpy as np
import poselib

RANSAC_THR, CONF = 2.5, 0.99999


def relative_pose_error(T_0to1, R, t):
    t_gt = T_0to1[:3, 3]; n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0))); t_err = np.minimum(t_err, 180 - t_err)
    cos = np.clip((np.trace(np.dot(R.T, T_0to1[:3, :3])) - 1) / 2, -1., 1.)
    return float(t_err), float(np.rad2deg(np.abs(np.arccos(cos))))


def camera(K):
    return {"model": "PINHOLE", "width": int(2 * K[0, 2]), "height": int(2 * K[1, 2]), "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}


def pose_error(pts0, pts1, K0, K1, T_0to1, seed):
    """Returns (max(R_err, t_err) in degrees, number of inliers). inf if no pose can be estimated."""
    if len(pts0) < 5: return np.inf, 0
    perm = np.random.default_rng(seed).permutation(len(pts0))
    try:
        M, info = poselib.estimate_relative_pose(pts0[perm].astype(np.float64), pts1[perm].astype(np.float64), camera(K0), camera(K1),
            {"max_epipolar_error": RANSAC_THR, "success_prob": CONF, "min_iterations": 20, "max_iterations": 1000, "seed": int(seed)})
    except Exception:
        return np.inf, 0
    t_err, R_err = relative_pose_error(T_0to1, M.R, M.t)
    if not np.isfinite(t_err) or not np.isfinite(R_err): return np.inf, 0
    return max(t_err, R_err), int(np.sum(info["inliers"]))


def error_auc(errors, thresholds=(5, 10, 20)):
    errors = [0] + sorted(list(errors)); recall = list(np.linspace(0, 1, len(errors))); out = {}
    for thr in thresholds:
        last = np.searchsorted(errors, thr); y = recall[:last] + [recall[last - 1]]; x = errors[:last] + [thr]
        out[f'auc@{thr}'] = float(np.trapz(y, x) / thr) if hasattr(np, 'trapz') else float(np.trapezoid(y, x) / thr)
    return out
