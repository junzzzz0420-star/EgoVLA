# -*- coding: utf-8 -*-
"""多目输入可视化：N 路输入视频（各带手部建模叠加）+ 1 个世界系双手 3D 面板。

布局由输入个数自动决定（这正是未来 UI 的三个开关 mono/stereo/rig6 的渲染端）：
    mono   → 1 输入面板 + 世界 3D
    stereo → 2 输入面板 + 世界 3D
    rig6   → 6 输入面板(3x2) + 世界 3D
世界 3D 的轨迹来自 --fused（单目=该目 parquet；多目=融合 parquet）并做 IMU 增强
（时间基自校准 + 重力对齐 + 去抖）；各输入面板的手部建模来自各目自身重建，
投影到各自画面（贴合检测）。

每次输出到 <out_root>/<tag>_<时间戳>/，不覆盖历史，便于对比。

用法:
    python viz_multiview.py --tag rig6 \
        --views camera0=<video0>,<parquet0> camera1=<video1>,<parquet1> ... \
        --fused <融合或参考 parquet> \
        --export <das_export/<rec>> --seg-start 2.40 --out-root examples/das_compare
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from ego_pipeline.tasks.lerobot_to_pi05 import derive_bimanual_ee
from ego_pipeline.tasks.visualize_pi05_ee import (
    _draw_ee_overlay, _draw_3d, _integrate_relative, _encode_mp4,
    _pad_to_height, LEFT_BGR, RIGHT_BGR)
from imu_enhance import compute_adjusted

VIEW_BG = (16, 24, 30)
GAP = 6


def load_view_state(parquet_path):
    """单目视图：derive 出该目首帧相机规范的 14 维 state + 逐帧 w2c/K。"""
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    w2c = np.stack([np.asarray(v).reshape(4, 4) for v in df['extrinsics_w2c']])
    K = np.asarray(df['intrinsics'].iloc[0], dtype=np.float64).reshape(3, 3)
    kept = {s: np.asarray(df[f'{s}_kept'].to_list(), bool) for s in ('left', 'right')}
    ee = derive_bimanual_ee(
        left_xyz=np.stack([np.asarray(v).reshape(3) for v in df['left_transl_world']]),
        right_xyz=np.stack([np.asarray(v).reshape(3) for v in df['right_transl_world']]),
        left_rot=np.stack([np.asarray(v).reshape(3, 3) for v in df['left_orient_world']]),
        right_rot=np.stack([np.asarray(v).reshape(3, 3) for v in df['right_orient_world']]),
        cam_R=w2c[:, :3, :3], cam_t=w2c[:, :3, 3],
        left_valid=kept['left'], right_valid=kept['right'])
    return {'state': np.asarray(ee['state'], np.float32),
            'action': np.asarray(ee['action'], np.float32),
            'valid': ee['hand_valid'],
            'cam': {'K': K, 'R': w2c[:, :3, :3], 't': w2c[:, :3, 3]}}


def read_frames(video):
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def grid_layout(n_views):
    """输入面板尺寸与世界面板宽度。返回 (view_w, view_h, cols, world_w, world_h)。"""
    if n_views <= 1:
        return 672, 378, 1, 1356, 520
    if n_views <= 2:
        return 672, 378, 2, 1356, 520
    if n_views <= 4:
        return 672, 378, 2, 1356, 520
    return 448, 252, 3, 1356, 520       # 6 目: 3 列 x 2 行


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--views', nargs='+', required=True,
                    help='cameraN=<video.mp4>,<parquet>')
    ap.add_argument('--fused', required=True, help='世界 3D 用的(融合)parquet')
    ap.add_argument('--export', required=True, help='das_export/<rec>（IMU）')
    ap.add_argument('--seg-start', type=float, required=True)
    ap.add_argument('--out-root', default='examples/das_compare')
    ap.add_argument('--tag', default='run')
    args = ap.parse_args()

    views = {}
    for spec in args.views:
        name, rest = spec.split('=', 1)
        video, parquet = rest.split(',', 1)
        views[name] = {'video': video, 'parquet': parquet}
    names = sorted(views)

    adj = compute_adjusted(args.fused, args.export, args.seg_start,
                           out='/tmp/viz_multiview_ctx')
    state_w = adj['states_after'].astype(np.float32)
    action_w = np.asarray(adj['res_a']['action'], np.float32)
    valid_w = adj['res_a']['hand_valid']
    recon_w = _integrate_relative(state_w, action_w)
    N = min(adj['N'], len(state_w))

    for nm in names:
        v = views[nm]
        v['frames'] = read_frames(v['video'])
        v['st'] = load_view_state(v['parquet'])
        N = min(N, len(v['frames']), len(v['st']['state']))

    vw, vh, cols, world_w, world_h = grid_layout(len(names))
    rows = (len(names) + cols - 1) // cols
    top_h = rows * vh + (rows - 1) * GAP
    canvas_w = max(cols * vw + (cols - 1) * GAP, world_w)
    canvas_h = top_h + GAP + world_h + 34

    out_dir = Path(args.out_root) / f"{args.tag}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[viz] {len(names)} 输入 + 世界3D, {N} 帧 → {out_dir}')

    title = f"{args.tag}  |  inputs={len(names)} ({','.join(names)})  |  world = IMU-enhanced fused  |  gravity-aligned"
    composed = []
    for t in range(N):
        canvas = np.full((canvas_h, canvas_w, 3), VIEW_BG, np.uint8)
        for i, nm in enumerate(names):
            v = views[nm]
            r, c = divmod(i, cols)
            x0, y0 = c * (vw + GAP), r * (vh + GAP)
            img = v['frames'][t].copy()
            st = v['st']
            _draw_ee_overlay(img, st['state'][t, :3], st['state'][t, 3:6],
                             st['state'][t, 6], bool(st['valid'][t, 0]),
                             LEFT_BGR, st['cam']['K'], st['cam'], t, tag='L')
            _draw_ee_overlay(img, st['state'][t, 7:10], st['state'][t, 10:13],
                             st['state'][t, 13], bool(st['valid'][t, 1]),
                             RIGHT_BGR, st['cam']['K'], st['cam'], t, tag='R')
            panel = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
            canvas[y0:y0 + vh, x0:x0 + vw] = panel
            cv2.putText(canvas, nm, (x0 + 10, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        world = _draw_3d(world_h, world_w, state_w[:N], recon_w, action_w, valid_w, t)
        y1 = top_h + GAP
        canvas[y1:y1 + world_h, 0:world_w] = world
        cv2.putText(canvas, title, (12, canvas_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 216), 1, cv2.LINE_AA)
        h, w = canvas.shape[:2]
        if h % 2 or w % 2:
            canvas = canvas[:h - h % 2, :w - w % 2]
        composed.append(canvas)
        if t % 25 == 0:
            print(f'[viz] {t}/{N}')

    mp4 = out_dir / 'multiview.mp4'
    _encode_mp4(composed, mp4, 15.0)
    (out_dir / 'snapshots').mkdir(exist_ok=True)
    for t in sorted({0, N // 3, 2 * N // 3, N - 1}):
        cv2.imwrite(str(out_dir / 'snapshots' / f'frame_{t:04d}.png'), composed[t])

    def jitter(s):
        dd = np.diff(np.diff(s[:, [0, 7]], axis=0), axis=0)
        return float(np.sqrt((dd ** 2).sum(-1)).mean())
    np.savez_compressed(out_dir / 'metrics.npz',
                        state=state_w[:N], jitter=jitter(state_w[:N]))
    print(f'[viz] EE 位置抖动 = {jitter(state_w[:N]):.5f}')
    print(f'[OK] {mp4}')


if __name__ == '__main__':
    main()
