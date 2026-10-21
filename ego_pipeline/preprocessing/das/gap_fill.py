#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fused.parquet 短时缺失补全 + 帧状态标记。

对每只手 kept=False 的连续缺失段：
  - 段长 ≤ --max-gap（默认 3 帧）且两端均有有效观测 → 保守插值补全，
    帧状态记 interpolated；
  - 其余（长缺失、段贴边无端点）→ 不生成数据，状态记 invalid；
  - 原有效帧 → observed。

插值方式：transl / kp21 线性；orient（3x3 矩阵展平）经 Slerp；
kp21_quat（21x4）逐元素线性后按关节归一化。补全帧 kept 置 True，但状态列
明确区分来源——补出的轨迹不与真实观测混同。

只读原文件，输出到 --out（默认 <input>_filled.parquet）。审计验证：
  python scripts/audit_trajectory.py --parquet <out> --out-dir <qc目录>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

HANDS = ('left', 'right')
# 列名 -> (维度, 插值方式)
COLS = {
    'transl_world':      (3,  'lerp'),
    'orient_world':      (9,  'slerp'),
    'hand_kp21_world':   (63, 'lerp'),
    'hand_kp21_quat_world': (84, 'qlerp'),
}
OBSERVED, INTERP, INVALID = 'observed', 'interpolated', 'invalid'


def _valid_row(df, hand, i):
    kept = bool(df[f'{hand}_kept'].iloc[i])
    t = df[f'{hand}_transl_world'].iloc[i]
    t = np.asarray(t, dtype=float) if t is not None else None
    ok = t is not None and t.size == 3 and np.all(np.isfinite(t))
    return kept and ok


def _gaps(n, valid_mask):
    """[(lo, hi)]：两端 valid、中间全 invalid 的开区间段。"""
    gaps, last = [], None
    for i in range(n):
        if valid_mask[i]:
            if last is not None and i - last > 1:
                gaps.append((last, i))
            last = i
    return gaps  # 首/尾贴边的缺失自动不含（last=None 或无闭合）


def _fill_col(kind, dim, vals, lo, hi):
    fill = np.arange(lo + 1, hi)
    t = np.array([lo, hi], dtype=float)
    if kind == 'lerp':
        out = np.empty((len(fill), dim))
        for d in range(dim):
            out[:, d] = np.interp(fill, t, vals[[lo, hi], d])
        return out
    if kind == 'slerp':                       # 3x3 矩阵展平
        r0 = Rotation.from_matrix(vals[lo].reshape(3, 3))
        r1 = Rotation.from_matrix(vals[hi].reshape(3, 3))
        return Slerp(t, Rotation.concatenate([r0, r1]))(fill).as_matrix().reshape(len(fill), 9)
    if kind == 'qlerp':                       # 21x4 关节四元数
        out = np.empty((len(fill), dim))
        for d in range(dim):
            out[:, d] = np.interp(fill, t, vals[[lo, hi], d])
        q = out.reshape(-1, 4)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        return q
    raise ValueError(kind)


def fill_parquet(path: Path, max_gap: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    n = len(df)
    stats = {}
    for h in HANDS:
        valid = np.array([_valid_row(df, h, i) for i in range(n)])
        status = np.array([OBSERVED if v else INVALID for v in valid], dtype=object)
        filled = 0
        cols = {c: (np.stack(df[f'{h}_{c}'].map(
                    lambda x: np.asarray(x, dtype=float))) if f'{h}_{c}' in df else None)
                for c in COLS}
        for lo, hi in _gaps(n, valid):
            if hi - lo - 1 > max_gap:
                continue
            for c, (dim, kind) in COLS.items():
                if cols[c] is None:
                    continue
                vals = cols[c]
                seg = _fill_col(kind, dim, vals, lo, hi)
                for k, i in enumerate(range(lo + 1, hi)):
                    df.at[i, f'{h}_{c}'] = seg[k]
            for i in range(lo + 1, hi):
                df.at[i, f'{h}_kept'] = True
                status[i] = INTERP
                filled += 1
        df[f'{h}_status'] = status
        stats[h] = {'observed': int(valid.sum()), 'interpolated': filled,
                    'invalid': int((status == INVALID).sum())}
    return df, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', default=None, help='默认 <input>_filled.parquet')
    ap.add_argument('--max-gap', type=int, default=3)
    args = ap.parse_args()

    src = Path(args.parquet)
    dst = Path(args.out) if args.out else src.with_name(src.stem + '_filled.parquet')
    df, stats = fill_parquet(src, args.max_gap)
    df.to_parquet(dst)
    print(f'[gap_fill] {src} -> {dst}  (max_gap={args.max_gap})')
    for h, s in stats.items():
        print(f'  {h}: {s}')


if __name__ == '__main__':
    main()
