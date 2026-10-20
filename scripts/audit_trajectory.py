#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹质量审计：读取融合 parquet（fused.parquet / das_compare 等），输出逐帧
质量 CSV 与汇总 JSON，只读不改原数据。

检测项（阈值均可 CLI 覆盖）：
  coverage      kept 帧比例 + transl/kp21 数值有效性
  missing_runs  连续 kept=False 的段（段数、最长段）
  suspect_speed 相邻有效帧速度 > --max-speed（默认 2.0 m/s）
  suspect_accel 速度二阶差分 > --max-accel（默认 20 m/s^2）
  lr_jump       双手间距单帧突变 > --lr-jump（默认 0.15 m，疑似身份互换/跳变）
  kp_deform     kp21 指尖-腕距/掌长 超出 [--deform-lo, --deform-hi]（默认 1.1~3.5）

用法：
  python scripts/audit_trajectory.py \
      --parquet 'examples/das_app_runs/*/fused.parquet' \
                examples/das_compare/stereo_fused.parquet \
      --out-dir examples/trajectory_qc
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HANDS = ('left', 'right')
KP_TIPS = (4, 8, 12, 16, 20)   # MANO21 指尖: thumb/index/middle/ring/pinky
KP_WRIST, KP_MCP_MIDDLE = 0, 9


def _vec(x):
    a = np.asarray(x, dtype=float)
    return a if a.size == 3 else None


def _finite(v):
    return v is not None and np.all(np.isfinite(v))


def audit(path: Path, args) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    n = len(df)
    ts = df['timestamp'].to_numpy(dtype=float) if 'timestamp' in df else None
    dts = np.diff(ts) if ts is not None and len(ts) > 1 else None

    rows, prev = [], {}   # prev: 有效帧的 (pos, t, speed) 用于跨缺失段差分
    for i in range(n):
        t = ts[i] if ts is not None else float(i) / args.fps
        rec = {'frame_index': int(df['frame_index'].iloc[i]) if 'frame_index' in df else i,
               'timestamp': t, 'flags': []}
        for h in HANDS:
            kept = bool(df[f'{h}_kept'].iloc[i]) if f'{h}_kept' in df else False
            pos = _vec(df[f'{h}_transl_world'].iloc[i]) if f'{h}_transl_world' in df else None
            valid = kept and _finite(pos)
            rec[f'{h}_kept'], rec[f'{h}_valid'] = kept, valid
            spd = acc = float('nan')
            if valid:
                if h in prev and prev[h] is not None:
                    dt = max(t - prev[h]['t'], 1e-3)
                    spd = float(np.linalg.norm(pos - prev[h]['pos']) / dt)
                    if prev[h]['spd'] == prev[h]['spd']:          # 非 NaN
                        acc = abs(spd - prev[h]['spd']) / dt
                prev[h] = {'pos': pos, 't': t, 'spd': spd}
                if spd > args.max_speed:
                    rec['flags'].append(f'suspect_speed_{h}')
                if acc == acc and acc > args.max_accel:
                    rec['flags'].append(f'suspect_accel_{h}')
                # kp21 形变（仅当列存在且有限）
                kpc = f'{h}_hand_kp21_world'
                if kpc in df.columns:
                    kp = df[kpc].iloc[i]
                    kp = np.asarray(kp, dtype=float).reshape(-1, 3) if kp is not None else None
                    if kp is None or kp.shape != (21, 3) or not np.all(np.isfinite(kp)):
                        rec['flags'].append(f'kp_invalid_{h}')
                    else:
                        palm = np.linalg.norm(kp[KP_MCP_MIDDLE] - kp[KP_WRIST])
                        if palm > 1e-6:
                            ratio = max(np.linalg.norm(kp[t_] - kp[KP_WRIST]) for t_ in KP_TIPS) / palm
                            if not (args.deform_lo <= ratio <= args.deform_hi):
                                rec['flags'].append(f'kp_deform_{h}')
            else:
                if kept:
                    rec['flags'].append(f'transl_invalid_{h}')
                else:
                    rec['flags'].append(f'missing_{h}')
                prev[h] = None
            rec[f'{h}_speed'], rec[f'{h}_accel'] = spd, acc
        # 双手间距突变
        if rec.get('left_valid') and rec.get('right_valid'):
            rec['lr_dist'] = float(np.linalg.norm(
                np.asarray(rec['_lp']) - np.asarray(rec['_rp']))) if '_lp' in rec else float('nan')
        rows.append(rec)

    # 第二遍补 lr_dist / lr_jump（避免在 rec 里塞临时向量）
    fdf = pd.DataFrame(rows)
    for i in range(n):
        lp = _vec(df[f'left_transl_world'].iloc[i]) if 'left_transl_world' in df else None
        rp = _vec(df[f'right_transl_world'].iloc[i]) if 'right_transl_world' in df else None
        fdf.at[i, 'lr_dist'] = float(np.linalg.norm(lp - rp)) if (_finite(lp) and _finite(rp)) else float('nan')
    d = fdf['lr_dist'].to_numpy(dtype=float)
    jumps = np.zeros(n, dtype=bool)
    idx = np.where(np.isfinite(d))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= args.lr_gap_frames and abs(d[b] - d[a]) > args.lr_jump:
            jumps[b] = True
    for i in np.where(jumps)[0]:
        fdf.at[i, 'flags'] = fdf.at[i, 'flags'] + ['lr_jump'] if isinstance(fdf.at[i, 'flags'], list) else ['lr_jump']
    fdf['flags'] = fdf['flags'].apply(lambda L: ';'.join(L))

    def _runs(mask):
        runs, cur = [], 0
        for m in mask:
            cur = cur + 1 if m else 0
            if cur and not m:
                pass
            runs.append(cur)
        # 计段
        segs, c = [], 0
        for m in mask:
            if m:
                c += 1
            elif c:
                segs.append(c); c = 0
        if c:
            segs.append(c)
        return len(segs), (max(segs) if segs else 0)

    summary = {'file': str(path), 'frames': n}
    for h in HANDS:
        kept = fdf[f'{h}_kept'].to_numpy(dtype=bool)
        valid = fdf[f'{h}_valid'].to_numpy(dtype=bool)
        nruns, longest = _runs(~kept)
        summary[f'{h}_coverage_pct'] = round(100 * kept.mean(), 1)
        summary[f'{h}_valid_pct'] = round(100 * valid.mean(), 1)
        summary[f'{h}_missing_runs'] = nruns
        summary[f'{h}_longest_missing'] = longest
        summary[f'{h}_suspect_speed'] = int(fdf['flags'].str.contains(f'suspect_speed_{h}').sum())
        summary[f'{h}_suspect_accel'] = int(fdf['flags'].str.contains(f'suspect_accel_{h}').sum())
        summary[f'{h}_kp_invalid'] = int(fdf['flags'].str.contains(f'kp_invalid_{h}').sum())
        summary[f'{h}_kp_deform'] = int(fdf['flags'].str.contains(f'kp_deform_{h}').sum())
    summary['lr_jump_frames'] = int(jumps.sum())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fdf.drop(columns=[c for c in fdf.columns if c.startswith('_')], errors='ignore').to_csv(
        Path(args.out_dir) / f'{path.parent.name}_frame_qc.csv', index=False)
    return fdf, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', nargs='+', required=True,
                    help='融合 parquet 路径或 glob（可多个）')
    ap.add_argument('--out-dir', default='examples/trajectory_qc')
    ap.add_argument('--max-speed', type=float, default=2.0, help='m/s，相邻有效帧速度阈值')
    ap.add_argument('--max-accel', type=float, default=20.0, help='m/s^2')
    ap.add_argument('--lr-jump', type=float, default=0.15, help='双手间距单帧突变阈值 m')
    ap.add_argument('--lr-gap-frames', type=int, default=3, help='间距检测允许的最大帧间隔')
    ap.add_argument('--deform-lo', type=float, default=1.1)
    ap.add_argument('--deform-hi', type=float, default=3.5)
    ap.add_argument('--fps', type=float, default=15.0, help='无 timestamp 列时的回退帧率')
    args = ap.parse_args()

    paths = []
    for pat in args.parquet:
        hit = sorted(globmod.glob(pat))
        paths += hit or ([pat] if Path(pat).exists() else [])
    if not paths:
        raise SystemExit(f'no parquet matched: {args.parquet}')

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for p in map(Path, paths):
        _, s = audit(p, args)
        summaries.append(s)
        print(f"[audit] {p}")
        print('  ' + json.dumps({k: v for k, v in s.items() if k != 'file'}, ensure_ascii=False))

    sm = pd.DataFrame(summaries)
    sm.to_json(out / 'summary.json', orient='records', force_ascii=False, indent=2)
    # 控制台汇总表（按最低覆盖排序）
    cov = sm[['file', 'frames', 'left_coverage_pct', 'right_coverage_pct',
              'left_longest_missing', 'right_longest_missing',
              'left_suspect_speed', 'right_suspect_speed', 'lr_jump_frames']]
    print('\n==== 汇总（按左手覆盖升序）====')
    print(cov.sort_values('left_coverage_pct').to_string(index=False))
    print(f'\n逐帧 CSV 与 summary.json 已写入 {out}/')


if __name__ == '__main__':
    main()
