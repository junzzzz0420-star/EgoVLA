#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.3 — HOI 融合审阅视频渲染：双目画面 + 轨迹条 + 语义注释框。

每个 clip 输出一个 review mp4（structured/hoi_fusion/review_videos/）：
  左半 CAM0 / 右半 CAM1 原始画面；顶部固定语义注释框（事件随时间切换，
  unknown/null 原样显示，NEEDS_REVIEW/CONFLICT 醒目标出，主事件与待复核
  事件分行区分）；底部双手腕部世界轨迹条（LEFT=绿 / RIGHT=橙，事件区间
  着色，当前位置打点）。
视频帧 ↔ 轨迹行按时间戳映射（Step 2.1 已验证的 round(t*15) 规则），
不假设帧号=行号。画面文字用英文（系统无 CJK 字体），JSON 数据保持原文。

用法：
  python ego_pipeline/preprocessing/das/render_hoi_review.py \
      --structured-dir examples/gapfill_lyz/egocap_20261001_103828/structured
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HAND_COLORS = {'left': (80, 220, 80), 'right': (60, 160, 255)}   # BGR
WARN = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_H = 240           # 注释框高度(px)：容纳双手并列各5行


def draw_box(img, lines):
    """左上固定注释框：半透明底 + 白/彩字。lines: [(text, color, scale)]"""
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1] - 4, BOX_H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    y = 22
    for text, color, scale in lines:
        cv2.putText(img, text, (10, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += 22 if scale > 0.45 else 18
    return img


def events_at(events, t):
    act = [e for e in events if e['start_time'] - 1e-6 <= t <= e['end_time'] + 1e-6]
    # 主事件 = 双手并列：left 优先，其次按开始时间；不再把另一只手降级为小字
    act.sort(key=lambda e: (e['hand_side'] != 'left', e['needs_review'],
                            -e['start_time']))
    return act[:2], act


def hand_lines(e):
    q = e.get('quality_status', {})
    return [
        (f"[{e['hand_side'].upper()}] obj={e['object_category']!r} "
         f"alt={e.get('object_category_alt') or '-'}", HAND_COLORS[e['hand_side']], 0.45),
        (f"  visual={e['visual_action']} prims={','.join(e['motion_primitives']) or 'none'}",
         HAND_COLORS[e['hand_side']], 0.45),
        (f"  semantic={e['semantic_action']} contact={e['contact_state']}", (255, 255, 255), 0.45),
        (f"  quality={q} status={e['verification_status']}", (255, 255, 255), 0.45),
    ]


def render_clip(clip, events, demo, wrists, clips_dir, out_path, det, fps_hint=15.0):
    caps = {}
    for key, ref_key in (('L', 'left_video_ref'), ('R', 'right_video_ref')):
        ref = clip.get(ref_key)
        p = clips_dir / Path(ref).name if ref else None
        caps[key] = cv2.VideoCapture(str(p)) if p and p.exists() else None
    if caps['L'] is None or not caps['L'].isOpened():
        raise FileNotFoundError(f'{clip["clip_id"]}: 左目 clip 视频不可读')
    n_frames = int(caps['L'].get(cv2.CAP_PROP_FRAME_COUNT))

    ts = demo['timestamp'].to_numpy(dtype=float)
    t0, t1 = clip['start_time'], clip['end_time']
    ev = [e for e in events if clip['clip_id'] in e['source_clip_ids']]
    allxy = np.concatenate([wrists[s][:, [0, 2]] for s in wrists])
    lo, hi = np.nanmin(allxy, 0), np.nanmax(allxy, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    okL, fL = caps['L'].read()
    okR, fR = (caps['R'].read() if caps['R'] else (False, None))
    H, W = fL.shape[:2]
    W2 = W * 2
    probe = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-s', f'{W2}x{H}', '-r', str(fps_hint), '-i', '-', '-c:v', 'libx264',
         '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(out_path)],
        stdin=subprocess.PIPE)

    def tag_hand(img, mains):
        """检测本目画面的手，画红框 + 框旁当前语义文字（cls0=左 cls1=右）。"""
        if det is None or img is None:
            return
        r = det.predict(img, verbose=False, conf=0.3, imgsz=640)[0]
        if r.boxes is None or not len(r.boxes):
            return
        xy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, c in zip(xy, cls):
            side = 'left' if c == 0 else 'right'
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            e = next((m for m in mains if m['hand_side'] == side), None)
            label = (f"{side.upper()}: {e['semantic_action']} | "
                     f"{e['object_category']}" + (" [REVIEW]" if e['needs_review'] else "")
                     ) if e else f"{side.upper()}: no active event"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
            ty = y0 - 6 if y0 - th - 6 > 0 else y1 + th + 6
            cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(img, label, (x0 + 3, ty), FONT, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    def overlay(k, fL, fR):
        fR_ = fR if (fR is not None and fR.shape == fL.shape) else np.zeros_like(fL)
        tag_hand(fL, mains_cur[0])
        if fR is not None:
            tag_hand(fR, mains_cur[1])
        frame = np.concatenate([fL, fR_], axis=1)
        t = t0 + k / fps_hint
        row = min(int(np.searchsorted(ts, t)), len(ts) - 1)
        mains, act = events_at(ev, t)
        lines = [(f"{clip['clip_id']}  t={t:.2f}s  frame={k}  (traj row {row})",
                  (255, 255, 255), 0.5)]
        if not mains:
            lines.append(('NO ACTIVE EVENT (between candidates)', (180, 180, 180), 0.5))
        for m in mains[:2]:                      # 双手同框时两只手并列完整显示
            lines.append((f"event={m['event_id']} " + (
                'NEEDS_REVIEW: ' + ','.join(m['conflicts'])
                if m['needs_review'] else 'no-conflict'),
                WARN if m['needs_review'] else (120, 255, 120), 0.45))
            lines += hand_lines(m)
            lines.append(('', (0, 0, 0), 0.3))
        for other in act[len(mains):]:           # 第三事件起仍保留灰字提示
            lines.append((f"(+more {other['event_id']} "
                          f"{other['hand_side']}/{str(other['object_category'])!r})",
                          (180, 180, 180), 0.42))
        draw_box(frame, [l for l in lines if l[0]][:11])
        cv2.putText(frame, 'CAM0 (left eye)', (8, H - 12), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'CAM1 (right eye)', (W2 // 2 + 8, H - 12), FONT, 0.5,
                    (255, 255, 255), 1)
        cv2.line(frame, (W2 // 2, 0), (W2 // 2, H), (255, 255, 255), 1)
        # 底部轨迹条：事件区带着色 + 双手腕世界(x,z)轨迹 + 当前点
        bar_y0, bar_h = H - 100, 90
        cv2.rectangle(frame, (0, bar_y0), (W2, H), (25, 25, 25), -1)
        for e in ev:
            x0 = int((e['start_time'] - t0) / max(t1 - t0, 1e-6) * W2)
            x1 = int((e['end_time'] - t0) / max(t1 - t0, 1e-6) * W2)
            cv2.rectangle(frame, (x0, bar_y0), (x1, bar_y0 + 8),
                          HAND_COLORS.get(e['hand_side'], (200, 200, 200)), -1)
        for side, arr in wrists.items():
            m = (ts >= t0 - 1e-6) & (ts <= t1 + 1e-6)
            pts = arr[m][:, [0, 2]]
            ok = np.isfinite(pts).all(1)
            xy = (pts - lo) / np.maximum(hi - lo, 1e-6)
            px = [(int(x * W2), bar_y0 + 10 + int((1 - y) * (bar_h - 20)))
                  for x, y in xy[ok]]
            for a, b in zip(px[:-1], px[1:]):
                cv2.line(frame, a, b, HAND_COLORS[side], 1)
            if px:
                idx = min(max(row - int(m.argmax()), 0), len(px) - 1)
                cv2.circle(frame, px[idx], 4, HAND_COLORS[side], -1)
        cv2.putText(frame, 'wrist world traj (x,z): LEFT green / RIGHT orange',
                    (8, bar_y0 + 14), FONT, 0.42, (200, 200, 200), 1)
        return frame

    k = 0
    while okL:
        t = t0 + k / fps_hint
        mains_cur = (events_at(ev, t)[0][:2], events_at(ev, t)[0][:2])
        probe.stdin.write(overlay(k, fL, fR).tobytes())
        k += 1
        okL, fL = caps['L'].read()
        okR, fR = (caps['R'].read() if caps['R'] else (False, None))
    for cap in caps.values():
        if cap is not None:
            cap.release()
    probe.stdin.close()
    probe.wait()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=True)
    args = ap.parse_args()
    sd = Path(args.structured_dir)
    out_dir = sd / 'hoi_fusion' / 'review_videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = pd.read_parquet(sd / 'demonstration.parquet')
    clips = json.loads((sd / 'visual_context/clip_manifest.json').read_text())
    ann = json.loads((sd / 'hoi_fusion/hoi_annotations.json').read_text())
    events = ann['events']
    clips_dir = sd / 'visual_context/clips'
    # 腕部世界坐标在 fused(_filled).parquet（demonstration 未含 transl 列）
    fused = next(p for p in (sd.parent / 'fused_filled.parquet',
                             sd.parent / 'fused.parquet') if p.exists())
    fdf = pd.read_parquet(fused)
    wrists = {s_: np.array([np.asarray(v, float) for v in fdf[f'{s_}_transl_world']])
              for s_ in ('left', 'right')}
    repo = Path(__file__).resolve().parents[3]
    det_path = repo / 'model/hawor/detector.pt'
    det = None
    if det_path.exists():
        from ultralytics import YOLO
        det = YOLO(str(det_path))
        det.to('cpu')
        print(f'[render] 手部检测器: {det_path} (cpu)')
    else:
        print('[render] 警告: 未找到 detector.pt，画面内不加手框')
    print(f'[render] 腕部轨迹源: {fused}')
    for clip in clips:
        p = render_clip(clip, events, demo, wrists, clips_dir,
                        out_dir / f'{clip["clip_id"]}_review.mp4', det)
        print(f'[render] {p}')
    print(f'[render] {len(clips)} 个审阅视频 -> {out_dir}')


if __name__ == '__main__':
    main()
