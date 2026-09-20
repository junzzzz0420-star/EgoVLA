"""Invalidate frames where the left wrist projects to the right of the right wrist.

Ego first-person hands should satisfy u_left < u_right in image space. Smooth
identity swaps (e.g. left track locked onto another person's hand on the right
side of the frame) pass kinematic outlier filters but violate this rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().copy()
    except Exception:
        pass
    return np.array(x, copy=True)


def _resolve_cam_K(cam_c2w, K):
    """Return (c2w (T,4,4), K (3,3)) or None if unusable."""
    if cam_c2w is None:
        return None
    if isinstance(cam_c2w, (str, Path)):
        p = Path(cam_c2w)
        if not p.exists():
            return None
        try:
            z = np.load(p)
            if "cam_c2w" not in z.files:
                return None
            c2w = np.asarray(z["cam_c2w"], dtype=np.float64)
            if K is None and "K" in z.files:
                K = np.asarray(z["K"], dtype=np.float64)
        except Exception:
            return None
    else:
        c2w = np.asarray(_np(cam_c2w), dtype=np.float64)

    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) < 1:
        return None
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return c2w, K


def _project_wrists(trans_world: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """Project world wrist positions to image u; return (u (2,T), depth (2,T), ok (2,T))."""
    T = min(trans_world.shape[1], len(c2w))
    u = np.full((2, T), np.nan, dtype=np.float64)
    depth = np.full((2, T), np.nan, dtype=np.float64)
    ok = np.zeros((2, T), dtype=bool)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for t in range(T):
        R = c2w[t, :3, :3]
        tw = c2w[t, :3, 3]
        R_w2c = R.T
        t_w2c = -R_w2c @ tw
        for h in (0, 1):
            pw = trans_world[h, t]
            pc = R_w2c @ pw + t_w2c
            z = float(pc[2])
            if not np.isfinite(z) or z <= 1e-4:
                continue
            u[h, t] = fx * float(pc[0]) / z + cx
            depth[h, t] = z
            ok[h, t] = True
    return u, depth, ok


def reject_crossed_hands(
    pred_result: list,
    cam_c2w=None,
    K=None,
    min_margin_px: float = 0.0,
    require_both: bool = True,
):
    """Mark frames invalid when left wrist image-u is not strictly left of right.

    Args:
        pred_result: [trans, rot, hand_pose, betas, valid]
        cam_c2w: (T,4,4) or path to megasam npz containing cam_c2w [/ K]
        K: (3,3) camera intrinsics; taken from npz if omitted
        min_margin_px: require u_right - u_left >= margin (0 = strict <)
        require_both: only enforce when both hands are currently valid
    """
    if cam_c2w is None:
        print("[clean] image_lr_constraint skipped (no cam_c2w)")
        return pred_result

    resolved = _resolve_cam_K(cam_c2w, K)
    if resolved is None:
        print("[clean] image_lr_constraint skipped (need cam_c2w + K)")
        return pred_result
    c2w, K = resolved

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result
    trans = _np(pred_trans)
    valid = _np(pred_valid).astype(bool)
    T = min(trans.shape[1], len(c2w), valid.shape[1])

    u, _depth, ok = _project_wrists(trans[:, :T], c2w[:T], K)

    both = valid[0, :T] & valid[1, :T] & ok[0, :T] & ok[1, :T]
    if require_both:
        enforce = both
    else:
        enforce = both  # same for now; single-hand frames have no L/R pair to compare

    crossed = enforce & ((u[1, :T] - u[0, :T]) < float(min_margin_px))
    n_cross = int(crossed.sum())
    if n_cross:
        # Identity is ambiguous once L/R order breaks — drop both hands.
        valid[0, :T][crossed] = False
        valid[1, :T][crossed] = False

    dtype = pred_valid.dtype if isinstance(pred_valid, np.ndarray) else np.float32
    print(
        f"[clean] image_lr_constraint crossed_frames={n_cross} "
        f"margin_px={min_margin_px} (invalidate both)"
    )
    return [trans, pred_rot, pred_hand_pose, pred_betas, valid.astype(dtype)]
