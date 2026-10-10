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

ACT_COLORS = {'hands_open': (46, 200, 90), 'hands_fist': (30, 150, 250),
              'hands_raise': (70, 170, 250), 'hands_lower': (220, 120, 230),
              'other': (150, 160, 168)}


def load_actions(json_path):
    """读 .actions.json → 逐帧标签 + 逐段 VLM 复核。返回 None 表示不可用。"""
    import json
    from label_actions import CN
    try:
        d = json.loads(Path(json_path).read_text(encoding='utf-8'))
        labels = d.get('labels') or []
        per_vlm = {}
        for s in d.get('segments', []):
            if s.get('vlm_label'):
                mark = '✓' if s.get('agree') else f'✗({CN[s["vlm_label"]]})'
                for t in range(s['start_frame'], s['end_frame'] + 1):
                    per_vlm[t] = mark
        segs = d.get('segments', [])
        per_open = {}
        for s in segs:
            if s.get('open_text'):
                for t in range(s['start_frame'], s['end_frame'] + 1):
                    per_open[t] = s['open_text']
        return {'labels': labels, 'cn': CN, 'per_vlm': per_vlm, 'per_open': per_open,
                'bars': [(s['start_frame'], s['end_frame'], s['label'])
                         for s in segs]}
    except Exception as e:
        print(f'[viz] 动作标注读取失败（画面不加标注条）: {e}')
        return None


def draw_semantic_strip(canvas, y0, h, w, t, N, acts, task_label, fonts):
    """顶部语义条：当前动作(彩色) + VLM 复核 + 任务 + 帧号，下面一条全程段轴游标。"""
    from PIL import Image, ImageDraw
    rgb = canvas[y0:y0 + h, 0:w][:, :, ::-1].copy()
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img, 'RGBA')
    f_big, f_sm = fonts

    lab = acts['labels'][t] if t < len(acts['labels']) else 'other'
    cn, col = acts['cn'][lab], ACT_COLORS.get(lab, ACT_COLORS['other'])
    col_rgb = col[::-1]
    x = 14
    dr.text((x, 4), f'动作: {cn}', font=f_big, fill=col_rgb)
    x += dr.textlength(f'动作: {cn}', font=f_big) + 24
    vlm = acts['per_vlm'].get(t)
    if vlm:
        v_col = (80, 220, 120) if vlm == '✓' else (70, 90, 240)
        dr.text((x, 10), f'VLM {vlm}', font=f_sm, fill=v_col)
        x += dr.textlength(f'VLM {vlm}', font=f_sm) + 24
    open_txt = acts.get('per_open', {}).get(t)
    y_task = 34 if open_txt else 10
    if task_label:
        tw = dr.textlength(f'任务: {task_label}', font=f_sm)
        dr.text((w - tw - 14 - 90, y_task), f'任务: {task_label}', font=f_sm,
                fill=(220, 226, 232))
    dr.text((w - 84, y_task), f'{t + 1}/{N} 帧', font=f_sm, fill=(160, 172, 182))
    if open_txt:
        dr.text((14, 34), f'描述: {open_txt}', font=f_sm, fill=(235, 220, 160))

    # 全程动作段轴：色块 = 检出段，白竖线 = 当前帧游标
    bx0, bx1, by = 14, w - 14, h - 18
    dr.rectangle([bx0, by, bx1, by + 12], fill=(30, 40, 48))
    for s_t0, s_t1, s_lab in acts.get('bars', []):
        x_a = bx0 + int((bx1 - bx0) * s_t0 / max(N - 1, 1))
        x_b = bx0 + int((bx1 - bx0) * s_t1 / max(N - 1, 1))
        c = ACT_COLORS.get(s_lab, ACT_COLORS['other'])[::-1]
        dr.rectangle([x_a, by, x_b, by + 12], fill=c + (210,))
        txt = acts['cn'][s_lab]
        if x_b - x_a > dr.textlength(txt, font=f_sm) + 16:
            dr.text((x_a + 8, by - 1), txt, font=f_sm, fill=(12, 16, 20))
    cx = bx0 + int((bx1 - bx0) * t / max(N - 1, 1))
    dr.line([cx, by - 5, cx, by + 17], fill=(255, 255, 255), width=2)
    canvas[y0:y0 + h, 0:w] = np.asarray(img)[:, :, ::-1]


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
    ap.add_argument('--actions', default=None,
                    help='label_actions.py 输出的 .actions.json（提供则把动作/VLM复核/任务烧进画面）')
    ap.add_argument('--task-label', default=None, help='任务语义文字（烧进画面右上）')
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

    # IMU 曲线（重力对齐后世界系；与 imu_enhanced 同源），随帧画游标烧进视频
    a_w_al = adj['a_w'] @ adj['R_align'].T
    w_w_al = adj['w_w'] @ adj['R_align'].T
    t_seg = adj['t_seg']
    t_imu_seg = (adj['t_mid'] - adj['t_k'][0]) / 1e9

    for nm in names:
        v = views[nm]
        v['frames'] = read_frames(v['video'])
        v['st'] = load_view_state(v['parquet'])
        N = min(N, len(v['frames']), len(v['st']['state']))

    vw, vh, cols, world_w, world_h = grid_layout(len(names))
    rows = (len(names) + cols - 1) // cols
    top_h = rows * vh + (rows - 1) * GAP
    imu_h = 250
    acts = load_actions(args.actions) if args.actions else None
    strip_h = 0 if acts is None else (78 if acts.get('per_open') else 56)
    canvas_w = max(cols * vw + (cols - 1) * GAP, world_w)
    canvas_h = top_h + GAP + world_h + GAP + imu_h + 34 + (strip_h + GAP if acts else 0)
    if acts is not None:
        from PIL import ImageFont
        try:
            fonts = (ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 24),
                     ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 18))
        except Exception:
            fonts = (ImageFont.load_default(), ImageFont.load_default())

    IMU_COLORS = ((25, 118, 210), (46, 160, 67), (224, 49, 49))

    def imu_row(k):
        """渲染当前帧的 IMU 双面板（matplotlib），返回 BGR 图。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        try:
            font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
            plt.rcParams['font.family'] = font_manager.FontProperties(
                fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        fig, axes = plt.subplots(1, 2, figsize=(canvas_w / 100, imu_h / 100), dpi=100)
        for ax, data, deg, title, ref in (
                (axes[0], w_w_al, True, '陀螺仪 ω（世界系, °/s）— 重力对齐 + 时间基校准', None),
                (axes[1], a_w_al, False, '加速度 a（世界系, m/s²）— az≈-9.8 即重力对齐', -9.80665)):
            plot = np.degrees(data) if deg else data
            for j, c in enumerate(IMU_COLORS):
                ax.plot(t_imu_seg, plot[:, j], color=np.array(c)[::-1] / 255.0,
                        lw=1.0, label='xyz'[j])
            if ref is not None:
                ax.axhline(ref, color='0.5', lw=0.8, ls=':')
            ax.axvline(t_seg[k], color='k', lw=1.2, ls='--')
            cur = np.stack([np.interp(t_seg[k], t_imu_seg, plot[:, j]) for j in range(3)])
            cur_str = '  '.join(f"{'xyz'[j]}={cur[j]:+6.1f}" for j in range(3))
            ax.set_title(f'{title}   当前 {cur_str}', fontsize=9, loc='left')
            ax.legend(loc='upper right', fontsize=8, ncol=3)
            ax.grid(alpha=0.3)
            ax.set_xlabel('窗口内秒', fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
        plt.close(fig)
        return buf

    out_dir = Path(args.out_root) / f"{args.tag}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[viz] {len(names)} 输入 + 世界3D'
          f'{" + 动作语义条" if acts is not None else ""}, {N} 帧 → {out_dir}')

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
        y1 = top_h + GAP + (strip_h + GAP if acts is not None else 0)
        canvas[y1:y1 + world_h, 0:world_w] = world
        if acts is not None:
            draw_semantic_strip(canvas, top_h + GAP, strip_h, canvas_w, t, N,
                                acts, args.task_label, fonts)
        imu = imu_row(t)
        ih, iw = imu.shape[:2]
        if iw >= canvas_w:
            canvas[y1 + world_h + GAP:y1 + world_h + GAP + ih, 0:canvas_w] = \
                imu[:ih, :canvas_w]
        else:
            resized = cv2.resize(imu, (canvas_w, imu_h), interpolation=cv2.INTER_AREA)
            canvas[y1 + world_h + GAP:y1 + world_h + GAP + imu_h, 0:canvas_w] = resized
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
                        state=state_w[:N], jitter=jitter(state_w[:N]),
                        imu_t=t_imu_seg[:N - 1] if len(t_imu_seg) >= N else t_imu_seg,
                        imu_gyro_w=w_w_al[:N - 1], imu_accel_w=a_w_al[:N - 1],
                        R_align=adj['R_align'])
    # 全窗 IMU 静态图（随产物保存，历史目录里可直接看）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), dpi=110)
        axes[0].plot(t_imu_seg, np.degrees(w_w_al))
        axes[0].set_title('陀螺仪 ω（世界系, °/s）'); axes[0].grid(alpha=.3)
        axes[1].plot(t_imu_seg, a_w_al)
        axes[1].axhline(-9.80665, color='0.5', ls=':')
        axes[1].set_title('加速度 a（世界系, m/s²）'); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / 'imu_curves.png', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[viz] imu_curves.png 保存失败: {e}')
    print(f'[viz] EE 位置抖动 = {jitter(state_w[:N]):.5f}')
    print(f'[OK] {mp4}')


if __name__ == '__main__':
    main()
