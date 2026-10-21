#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""结构化动作标注（第一阶段）：3D 手部运动学 → 动作原语候选 + 结构化中间表示。

输入 run_multiview/fuse 产出的 fused(_filled).parquet，输出：
  demonstration.parquet  逐帧特征 + 逐手原语 + 质量状态
  action_segments.json   事件级分段（逐手段 + 双手协同事件）+ 完整元数据

设计约束（勿破坏）：
  - 独立于 label_actions.py，不修改现有管线；
  - 全部差分使用实际 timestamp，不假设固定帧率；
  - 左右手独立计算，绝不合并均值；
  - 质量语义透传：observed=实测 / interpolated=gap_fill 补的(合成) / invalid=缺失；
  - 原语是**规则候选**，不是经过验证的物体交互；object/contact 一律 null，
    由后续 Qwen2.5-VL 接口（见 vlm_review 占位）填充，绝不伪造物理观测。

用法：
  python ego_pipeline/preprocessing/das/label_structured.py \
      --parquet examples/gapfill_lyz/egocap_20261001_103828/fused_filled.parquet
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

SIDES = ('left', 'right')
TIP_IDS = (4, 8, 12, 16, 20)          # 拇/食/中/无名/小 指尖
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')

# 物理量纲默认阈值（CLI 可覆盖）
DEFAULTS = dict(
    pinch_lo=0.35,        # 捏取指数阈值（拇指-食指尖距/掌长）
    fist_ext=1.05,        # 五指伸展中位数低于此 → 收拢
    open_ext=1.40,        # 张开判定（用于 release 事件）
    speed_reach=0.35,     # m/s，平移速度高于此 → reach
    speed_transport=0.08, # m/s，持有状态下的移动阈值
    omega_rotate=1.5,     # rad/s，腕部角速度高于此且位置稳定 → rotate
    min_seg_s=0.25,       # 段最短时长（秒）
    smooth_s=0.15,        # 特征平滑时间窗（秒）
)


# ---------------------------------------------------------------- 数据读取

def load_sources(parquet_path: Path):
    """读取 fused(_filled).parquet 的实际字段；缺必需列立即报错。"""
    df = pq.read_table(parquet_path).to_pandas()
    need = ['timestamp'] + [f'{s}_{c}' for s in SIDES
                            for c in ('kept', 'transl_world', 'orient_world',
                                      'hand_kp21_world')]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f'{parquet_path}: 缺少必需列 {missing}（请用 fuse_multiview '
                         f'产物，或先运行 gap_fill 得到 *_filled.parquet）')
    if len(df) < 2:
        raise ValueError(f'{parquet_path}: 仅 {len(df)} 行，无法差分')
    ts = df['timestamp'].to_numpy(dtype=float)
    if not np.all(np.diff(ts) > 0):
        raise ValueError(f'{parquet_path}: timestamp 非单调递增')
    hands = {}
    for s in SIDES:
        hands[s] = {
            'kp': np.stack([np.asarray(v, dtype=float).reshape(21, 3)
                            for v in df[f'{s}_hand_kp21_world']]),
            'wrist': np.stack([np.asarray(v, dtype=float).reshape(3)
                               for v in df[f'{s}_transl_world']]),
            'R': np.stack([np.asarray(v, dtype=float).reshape(3, 3)
                           for v in df[f'{s}_orient_world']]),
            'kept': np.asarray(df[f'{s}_kept'].to_list(), bool),
            # *_status 仅 filled 版存在；原始版由 kept 推导
            'status': (df[f'{s}_status'].tolist()
                       if f'{s}_status' in df.columns else None),
        }
    return df, ts, hands


def status_of(hand) -> np.ndarray:
    """observed / interpolated / invalid。"""
    if hand['status'] is not None:
        return np.asarray(hand['status'], dtype=object)
    return np.where(hand['kept'], 'observed', 'invalid')


# ---------------------------------------------------------------- 逐手特征

def hand_features(hand, ts, win_s):
    """全部用实际 timestamp 差分；无效帧输出 NaN。左右手各自独立调用。"""
    n = len(ts)
    kp, wrist = hand['kp'], hand['wrist']
    valid = hand['kept'] & np.all(np.isfinite(wrist), axis=1) \
        & np.all(np.isfinite(kp.reshape(n, -1)), axis=1)

    palm = np.linalg.norm(kp[:, 9] - kp[:, 0], axis=-1) + 1e-9
    ext = np.full((n, 5), np.nan)
    for j, tip in enumerate(TIP_IDS):
        ext[:, j] = np.linalg.norm(kp[:, tip] - kp[:, 0], axis=-1) / palm
    pinch = np.linalg.norm(kp[:, 4] - kp[:, 8], axis=-1) / palm
    ext[~valid] = np.nan
    pinch[~valid] = np.nan

    dt = np.diff(ts)
    trans = np.full(n, np.nan)
    trans[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / dt
    trans[1:][~(valid[:-1] & valid[1:])] = np.nan   # 跨无效帧差分不可信

    ang = np.full(n, np.nan)
    ok2 = valid[:-1] & valid[1:]
    if ok2.any():
        r0 = Rotation.from_matrix(hand['R'][:-1][ok2])
        r1 = Rotation.from_matrix(hand['R'][1:][ok2])
        ang[1:][ok2] = (r0.inv() * r1).magnitude() / dt[ok2]

    def smooth(x):
        k = max(1, int(round(win_s / max(float(np.median(dt)), 1e-3))))
        if k <= 1:
            return x
        half = k // 2
        out = np.empty_like(x)
        for i in range(n):
            seg = x[max(0, i - half):i + half + 1]
            out[i] = np.nanmedian(seg) if np.any(np.isfinite(seg)) else np.nan
        return out

    return {'valid': valid, 'wrist': wrist, 'ext': ext, 'pinch': smooth(pinch),
            'trans': smooth(trans), 'ang': smooth(ang)}


# ---------------------------------------------------------------- 原语状态机

def classify_primitives(f, th):
    """逐帧原语（规则候选）。持有(transport/hold/rotate)由时序状态延续。"""
    n = len(f['pinch'])
    prim = np.array(['unknown'] * n, dtype=object)
    with warnings.catch_warnings():                 # 全 NaN 行(该手整段无效)属预期
        warnings.simplefilter('ignore', RuntimeWarning)
        ext_med = np.nanmedian(f['ext'], axis=1)
    held, prev_pinch = None, np.nan
    for i in range(n):
        if not f['valid'][i]:
            held = None
            prev_pinch = np.nan
            continue
        pinch_now = f['pinch'][i] < th['pinch_lo']
        power_now = ext_med[i] < th['fist_ext']
        rotating = (np.isfinite(f['ang'][i]) and f['ang'][i] > th['omega_rotate']
                    and np.isfinite(f['trans'][i]) and f['trans'][i] < th['speed_reach'])
        if pinch_now or power_now:
            prim[i] = 'pinch' if pinch_now else 'power_grasp'
            held = prim[i]
        elif held is not None:
            opened = (held == 'pinch' and np.isfinite(prev_pinch)
                      and prev_pinch < th['pinch_lo'] + 0.10
                      and f['pinch'][i] > th['open_ext'] * 0.8) or \
                     (held == 'power_grasp' and ext_med[i] > th['open_ext'])
            if opened:
                prim[i] = 'release'
                held = None
            elif rotating:
                prim[i] = 'rotate'
            elif f['trans'][i] > th['speed_transport']:
                prim[i] = 'transport'
            else:
                prim[i] = 'hold'
        else:
            if rotating:
                prim[i] = 'rotate'
            elif f['trans'][i] > th['speed_reach']:
                prim[i] = 'reach'
            else:
                prim[i] = 'rest'
        prev_pinch = f['pinch'][i]
    return prim


# ---------------------------------------------------------------- 分段与协同

def segmentate(prim, ts, min_s):
    """按实际 timestamp 分段；unknown/rest 不输出（非事件）。"""
    segs, start = [], 0
    for i in range(1, len(prim) + 1):
        if i == len(prim) or prim[i] != prim[start]:
            if prim[start] not in ('unknown', 'rest') \
                    and ts[i - 1] - ts[start] >= min_s:
                segs.append({'primitive': str(prim[start]),
                             'start_frame': int(start), 'end_frame': int(i - 1),
                             'start_ts': round(float(ts[start]), 3),
                             'end_ts': round(float(ts[i - 1]), 3),
                             'duration_s': round(float(ts[i - 1] - ts[start]), 3)})
            start = i
    return segs


def bimanual_events(prims, feats, ts, min_s):
    """双手协同：同窗 rotate / 同窗移动且双手间距稳定。"""
    n = len(ts)
    both_rot = (prims['left'] == 'rotate') & (prims['right'] == 'rotate')
    mv = lambda s: np.array([p in ('transport', 'reach') for p in prims[s]])
    both_move = mv('left') & mv('right')

    d = np.linalg.norm(feats['left']['wrist'] - feats['right']['wrist'], axis=1)
    d_speed = np.full(n, np.nan)
    d_speed[1:] = np.abs(np.diff(d)) / np.diff(ts)
    stable = np.nan_to_num(d_speed, nan=9.0) < 0.10

    ev = []
    for mask, kind in ((both_rot, 'bimanual_rotate'),
                       (both_move & stable & ~both_rot, 'bimanual_transport')):
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        for chunk in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(chunk) >= 2 and ts[chunk[-1]] - ts[chunk[0]] >= min_s:
                ev.append({'event': kind,
                           'start_frame': int(chunk[0]), 'end_frame': int(chunk[-1]),
                           'start_ts': round(float(ts[chunk[0]]), 3),
                           'end_ts': round(float(ts[chunk[-1]]), 3),
                           'duration_s': round(float(ts[chunk[-1]] - ts[chunk[0]]), 3)})
    return ev


# ---------------------------------------------------------------- VLM 预留接口

def vlm_review(action_segments: dict, video_path: Path, model_id: str = None) -> dict:
    """预留接口（本阶段不实现、不引入新模型依赖）。

    契约：输入本模块的 action_segments.json 与对应窗口视频，输出按
    (start_frame, end_frame, hand) 对齐的 object / contact / review 字段，
    回填各段的 vlm 键；实现方不得修改帧边界与质量状态。
    """
    raise NotImplementedError('Qwen2.5-VL 物体识别与交互语义融合在第二阶段接入')


# ---------------------------------------------------------------- 主流程

def build(parquet_path: Path, out_dir: Path, th):
    df, ts, hands = load_sources(parquet_path)
    n = len(df)
    feats, prims = {}, {}
    demo = {'frame_index': (df['frame_index'].tolist()
                            if 'frame_index' in df else list(range(n))),
            'timestamp': ts.tolist()}
    for s in SIDES:
        feats[s] = hand_features(hands[s], ts, th['smooth_s'])
        prims[s] = classify_primitives(feats[s], th)
        demo[f'{s}_status'] = status_of(hands[s])
        demo[f'{s}_pinch_idx'] = np.round(feats[s]['pinch'], 4).tolist()
        for j, fn in enumerate(FINGERS):
            demo[f'{s}_ext_{fn}'] = np.round(feats[s]['ext'][:, j], 4).tolist()
        demo[f'{s}_trans_speed'] = np.round(feats[s]['trans'], 4).tolist()
        demo[f'{s}_ang_speed'] = np.round(feats[s]['ang'], 4).tolist()
        demo[f'{s}_primitive'] = prims[s]
    demo['hands_dist_m'] = np.round(
        np.linalg.norm(hands['left']['wrist'] - hands['right']['wrist'], axis=1), 4).tolist()

    ev = bimanual_events(prims, feats, ts, th['min_seg_s'])

    out_dir.mkdir(parents=True, exist_ok=True)
    demo_path = out_dir / 'demonstration.parquet'
    pd.DataFrame(demo).to_parquet(demo_path, index=False)

    seg_json = {
        'meta': {
            'source_parquet': str(parquet_path),
            'frames': n,
            'duration_s': round(float(ts[-1] - ts[0]), 3),
            'coordinate_system':
                'world frame of the reference view (camera0 first-frame '
                'convention); gravity-aligned when IMU enhancement ran',
            'units': {'translation': 'm', 'speed': 'm/s',
                      'angular_speed': 'rad/s', 'timestamp': 's (relative)'},
            'pose_representation':
                'wrist: 3x3 rotation matrix (source *_orient_world, flattened '
                '9); joints: kp21 xyz in meters (*_hand_kp21_world)',
            'quality_semantics': {
                'observed': 'measured by the reconstruction pipeline',
                'interpolated': 'synthesized by gap_fill (<=3-frame linear/'
                                'slerp fill); NOT a physical observation',
                'invalid': 'missing / unusable'},
            'primitive_semantics':
                'rule-based CANDIDATES from 3D hand kinematics only; NOT '
                'verified object interactions. object/contact are null by '
                'design until the VLM stage fills them.',
            'thresholds': dict(th),
        },
        'hands': {s: segmentate(prims[s], ts, th['min_seg_s']) for s in SIDES},
        'bimanual_events': ev,
        'vlm': {'status': 'reserved', 'object': None, 'contact': None,
                'review': None, 'interface': 'label_structured.vlm_review()'},
    }
    seg_path = out_dir / 'action_segments.json'
    seg_path.write_text(json.dumps(seg_json, ensure_ascii=False, indent=2))
    return demo_path, seg_path, seg_json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True, help='fused(_filled).parquet')
    ap.add_argument('--out-dir', default=None, help='默认输入同目录 structured/')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k.replace("_", "-")}', type=float, default=v)
    args = ap.parse_args()
    th = {k: getattr(args, k) for k in DEFAULTS}

    src = Path(args.parquet)
    out = Path(args.out_dir) if args.out_dir else src.parent / 'structured'
    demo_path, seg_path, seg = build(src, out, th)
    print(f'[structured] {src} ->\n  {demo_path}\n  {seg_path}')
    d = pd.read_parquet(demo_path)
    for s in SIDES:
        print(f'  {s} 原语分布: {dict(Counter(d[f"{s}_primitive"]))}')
    print(f'  段: 左 {len(seg["hands"]["left"])} / 右 {len(seg["hands"]["right"])}，'
          f'双手协同事件 {len(seg["bimanual_events"])} 个')


if __name__ == '__main__':
    main()
