#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.5 — 手-物体关联与交互验证（grounding）。

用两个现成检测器（零训练、零标签泄漏）在**转换件高清帧**上建立视觉证据：
  · WiLoR hand detector（cls0=左手, cls1=右手）→ 逐帧手 bbox
  · COCO yolo11n + ByteTrack → 物体 bbox + track_id（mouse/keyboard/cell
    phone/remote 等 8 类常见桌面物体）
坐标一律归一化到 [0,1]（跨分辨率安全）。帧索引 = (crop_offset + t) * 30，
即 Step 2.1 已验证时间映射的逆用；输出统一带窗口全局时间戳 t。

输出 structured/grounding/：
  hand_boxes.parquet        逐帧手 bbox（camera/hand/frame_t/norm bbox/conf）
  object_tracks.parquet     逐帧物体（camera/track_id/cls/conf/bbox/frame_t）
  association_evidence.parquet  逐帧×手×物体：中心距/IoU/速度相关/持续接近
  grounding_report.json     时间映射自检 + 物体检出统计

只读原始证据；关联证据供 fuse_hoi 仲裁，本模块不修改任何 Step 2.1-2.4 输出。

用法：
  python ego_pipeline/preprocessing/das/ground_objects.py \
      --structured-dir examples/gapfill_lyz/egocap_20261001_103828/structured \
      --export examples/egocap_das_export/egocap_20261001_103828_bag_0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COCO_KEEP = {'mouse': 'mouse', 'keyboard': 'keyboard', 'cell phone': 'cell phone',
             'remote': 'remote', 'book': 'book', 'cup': 'cup',
             'scissors': 'scissors', 'laptop': 'laptop'}
# VLM 中文/别名 → COCO 归一名（用于类别一致性比对）
ALIAS = {'鼠标': 'mouse', '电脑鼠标': 'mouse', 'computer mouse': 'mouse', 'mouse': 'mouse',
         '手机': 'cell phone', 'cell phone': 'cell phone', 'phone': 'cell phone',
         '键盘': 'keyboard', 'keyboard': 'keyboard',
         '游戏手柄': 'remote', '手柄': 'remote', 'gamepad': 'remote', 'remote': 'remote'}


def norm_alias(name: str | None):
    return ALIAS.get((name or '').strip().lower(), None)


def crop_offset(video_name: str) -> float:
    m = re.search(r'_rig_([\d.]+)s_', video_name)
    return float(m.group(1)) if m else 0.0


def detect_camera(video_hd: Path, camera_id: str, hand_model, obj_model,
                  t0: float, t1: float, stride: int = 2):
    """在 HD 帧上检测手与物体（30fps 每 stride 帧取一）。返回两个行列表。"""
    import cv2
    cap = cv2.VideoCapture(str(video_hd))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hands, objs = [], []
    for fi in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        t = fi / fps - _offset_for(video_hd)
        if t < t0 - 0.2 or t > t1 + 0.2:
            continue
        rh = hand_model.predict(frame, verbose=False, conf=0.3, imgsz=640)[0]
        if rh.boxes is not None and len(rh.boxes):
            for b, c, cf in zip(rh.boxes.xyxy.cpu().numpy(),
                                rh.boxes.cls.cpu().numpy().astype(int),
                                rh.boxes.conf.cpu().numpy()):
                hands.append({'camera': camera_id,
                              'hand': 'left' if c == 0 else 'right',
                              'frame_t': round(t, 3), 'hd_frame': fi,
                              'x0': b[0] / W, 'y0': b[1] / H,
                              'x1': b[2] / W, 'y1': b[3] / H,
                              'conf': round(float(cf), 3)})
        ro = obj_model.track(frame, persist=True, verbose=False, conf=0.22,
                             imgsz=800, classes=[i for i, n in
                                                 obj_model.names.items()
                                                 if n in COCO_KEEP])[0]
        if ro.boxes is not None and len(ro.boxes):
            ids = ro.boxes.id.int().cpu().numpy() if ro.boxes.id is not None \
                else [-1] * len(ro.boxes)
            for b, c, tid, cf in zip(ro.boxes.xyxy.cpu().numpy(),
                                     ro.boxes.cls.cpu().numpy().astype(int),
                                     ids, ro.boxes.conf.cpu().numpy()):
                objs.append({'camera': camera_id,
                             'cls': obj_model.names[c],
                             'track_id': int(tid), 'frame_t': round(t, 3),
                             'hd_frame': fi,
                             'x0': b[0] / W, 'y0': b[1] / H,
                             'x1': b[2] / W, 'y1': b[3] / H,
                             'conf': round(float(cf), 3)})
    cap.release()
    return hands, objs


_offset = {'val': 0.0}
def _offset_for(video_hd: Path) -> float:
    """HD 转换件 0 点 = 窗口时间 0 点前 crop_offset 秒（Step 2.1 验证映射）。"""
    return _offset['val']


def associate(hands: pd.DataFrame, objs: pd.DataFrame, win: float = 0.35):
    """逐帧×手×物体证据：归一中心距、IoU、窗口速度相关、持续接近计数。"""
    rows = []
    if hands.empty or objs.empty:
        return pd.DataFrame(columns=['frame_t', 'hand', 'cls', 'track_id', 'dist',
                                     'iou', 'move_corr', 'near_frames', 'camera'])
    for (cam, t), hg in hands.groupby(['camera', 'frame_t']):
        og = objs[(objs.camera == cam) & (np.abs(objs.frame_t - t) < 1e-6)]
        for _, o in og.iterrows():
            for _, h in hg.iterrows():
                hx, hy = (h.x0 + h.x1) / 2, (h.y0 + h.y1) / 2
                ox, oy = (o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2
                diag = np.hypot(h.x1 - h.x0, h.y1 - h.y0) + 1e-6
                dist = np.hypot(hx - ox, hy - oy) / diag
                ix0, iy0 = max(h.x0, o.x0), max(h.y0, o.y0)
                ix1, iy1 = min(h.x1, o.x1), min(h.y1, o.y1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = (h.x1 - h.x0) * (h.y1 - h.y0) + \
                    (o.x1 - o.x0) * (o.y1 - o.y0) - inter
                # 近窗速度相关（物体跟手走=持握证据）
                hw = hands[(hands.camera == cam) & (hands.hand == h.hand) &
                           (np.abs(hands.frame_t - t) < win)]
                ow = objs[(objs.camera == cam) & (objs.track_id == o.track_id) &
                          (np.abs(objs.frame_t - t) < win)].sort_values('frame_t')
                hw = hw.sort_values('frame_t')
                corr = np.nan
                if len(hw) >= 4 and len(ow) >= 4:
                    hv = np.diff(np.stack([(hw.x0 + hw.x1) / 2,
                                           (hw.y0 + hw.y1) / 2], 1), axis=0)
                    ov = np.diff(np.stack([(ow.x0 + ow.x1) / 2,
                                           (ow.y0 + ow.y1) / 2], 1), axis=0)
                    k = min(len(hv), len(ov), 6)
                    if k >= 3:
                        hv, ov = hv[-k:].ravel(), ov[-k:].ravel()
                        if hv.std() > 1e-6 and ov.std() > 1e-6:
                            corr = round(float(np.corrcoef(hv, ov)[0, 1]), 3)
                # 该手与此 track 在近窗内“中心距<1.0 手对角线”的帧数
                hw2 = hands[(hands.camera == cam) & (hands.hand == h.hand)]
                ow2 = objs[(objs.camera == cam) & (objs.track_id == o.track_id)]
                m2 = hw2.merge(ow2, on='frame_t', suffixes=('_h', '_o'))
                if len(m2):
                    d2 = np.hypot((m2.x0_h + m2.x1_h) / 2 - (m2.x0_o + m2.x1_o) / 2,
                                  (m2.y0_h + m2.y1_h) / 2 - (m2.y0_o + m2.y1_o) / 2)
                    diag2 = np.hypot(m2.x1_h - m2.x0_h, m2.y1_h - m2.y0_h) + 1e-6
                    near_n = int(((d2 / diag2) < 1.0).sum())
                else:
                    near_n = 0
                rows.append({'camera': cam, 'frame_t': t, 'hand': h.hand,
                             'cls': o.cls, 'track_id': o.track_id,
                             'dist': round(float(dist), 3),
                             'iou': round(float(inter / (union + 1e-6)), 3),
                             'move_corr': corr, 'near_frames': near_n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=True)
    ap.add_argument('--export', required=True, help='转换件目录（HD 源）')
    ap.add_argument('--stride', type=int, default=2, help='HD 30fps 每 N 帧检测一次')
    args = ap.parse_args()
    sd = Path(args.structured_dir)
    out = sd / 'grounding'
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    repo = Path(__file__).resolve().parents[3]
    hand_m = YOLO(str(repo / 'model/hawor/detector.pt'))
    obj_m = YOLO('yolo11n.pt')          # COCO 预训练，首次自动下载
    hand_m.to('cuda'); obj_m.to('cuda')

    align = json.loads((sd / 'visual_context/video_alignment.json').read_text())
    export = Path(args.export)
    # rig 文件名给出窗口起点；HD 帧率实测
    rig_l = align['videos']['left']['path']
    offset = crop_offset(Path(rig_l).name)
    _offset['val'] = offset

    import cv2
    cap = cv2.VideoCapture(str(export / 'camera0.mp4'))
    hd_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_hd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    t_end = min(offset + n_hd / hd_fps, 60.0) - offset

    # 时间映射自检：帧 t ↔ HD 帧 round((offset+t)*hd_fps) 必须落在 HD 范围内
    ts_check = json.loads((sd / 'visual_context/clip_manifest.json').read_text())
    map_ok = all(round((offset + c['end_time']) * hd_fps) <= n_hd + 1 for c in ts_check)

    all_h, all_o = [], []
    for cam_i, key in ((0, 'L'), (1, 'R')):
        h, o = detect_camera(export / f'camera{cam_i}.mp4', key, hand_m, obj_m,
                             0.0, t_end, stride=args.stride)
        all_h += h
        all_o += o
        print(f'[ground] camera{cam_i}: 手 {len(h)} 帧×只, 物体 {len(o)} 检出')
    hands = pd.DataFrame(all_h)
    objs = pd.DataFrame(all_o)
    hands.to_parquet(out / 'hand_boxes.parquet', index=False)
    objs.to_parquet(out / 'object_tracks.parquet', index=False)

    assoc = associate(hands, objs)
    assoc.to_parquet(out / 'association_evidence.parquet', index=False)

    stat = {'time_mapping_selfcheck': {'crop_offset_s': offset, 'hd_fps': hd_fps,
                                       'hd_frames': n_hd, 'clips_within_hd': map_ok},
            'hand_frames': len(hands),
            'object_detections': objs.cls.value_counts().to_dict()
            if len(objs) else {},
            'association_rows': len(assoc)}
    (out / 'grounding_report.json').write_text(
        json.dumps(stat, ensure_ascii=False, indent=2))
    print('[ground] 统计:', json.dumps(stat, ensure_ascii=False)[:300])
    print(f'[ground] -> {out}')


if __name__ == '__main__':
    main()
