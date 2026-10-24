#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.3 — 多模态手-物交互融合：视觉语义候选 × 3D 运动轨迹/原语。

输入（只读）：structured/ 下 demonstration.parquet、action_segments.json、
visual_context/clip_manifest.json、visual_semantics/visual_predictions.json。
输出：structured/hoi_fusion/
  hoi_annotations.json    事件级 HOI 候选（两套时间边界、冲突、质量全保留）
  hoi_frame_labels.parquet 逐帧语义与轨迹引用
  fusion_report.json      融合统计
  review_queue.json       待人工确认事件与证据引用

原则：不加新模型；相机≠左右手；VLM 不得覆盖 3D 边界（不一致时两套并存标
review）；原语只是辅助证据（pinch≠成功抓取）；无跨片段物体跟踪则只给
clip 级局部物体标识；quality(observed/interpolated/invalid) 逐帧透传；
verification_status 恒为 pending_review，绝不默认 verified。

用法：
  python ego_pipeline/preprocessing/das/fuse_hoi.py \
      --structured-dir examples/gapfill_lyz/egocap_20261001_103828/structured
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 'hoi-v1'
FUSION_RULES_VERSION = 'v1+v2experimental(detector-first, GT 0/8 — 默认仍为保守仲裁)'
TIME_UNC = ('+-16.7 ms video resampling (Step 2.1 verified mapping); '
            'vlm_time_estimate itself is a coarse model guess')
BOUNDARY_DIFF_S = 0.3     # 视觉/3D 边界差异超过此值记冲突
INVALID_REVIEW_PCT = 0.2  # 事件帧 invalid 占比超过此值 → low_quality


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


POSE_CN = {'OPEN': '张开', 'CLOSED': '收拢', 'PINCH': '捏合',
           'HALF_OPEN': '半开', 'UNKNOWN': '未知'}
INTER_CN = {'NO_INTERACTION': '无交互', 'APPROACH': '接近', 'GRASP': '抓取',
            'HOLD': '持握', 'RELEASE': '释放'}
OBJECT_CN = {'cell phone': '手机', 'mouse': '鼠标', 'keyboard': '键盘',
             'remote': '手柄', 'book': '书', 'cup': '杯子',
             'scissors': '剪刀', 'laptop': '笔记本电脑'}
PHONE_ALIASES = {'cell phone', 'phone', 'mobile phone', 'smartphone', 'cellphone', '手机'}
INTERACTION_CN = {
    'pick_up_phone': '拿起手机',
    'hold_phone': '持握手机',
    'put_down_phone': '放下手机',
}


def pose_state_series(demo, hand):
    """逐帧手势姿态（与交互事件独立的通道）。证据不足 -> UNKNOWN。"""
    ext = demo[[f'{hand}_ext_{f}' for f in
                ('thumb', 'index', 'middle', 'ring', 'pinky')]].to_numpy(float)
    ext_med = np.nanmedian(ext, axis=1)
    pinch = demo[f'{hand}_pinch_idx'].to_numpy(float)
    valid = demo[f'{hand}_status'].to_numpy() != 'invalid'
    st = np.array(['UNKNOWN'] * len(demo), dtype=object)
    for i in range(len(demo)):
        if not valid[i] or not np.isfinite(ext_med[i]):
            continue
        if np.isfinite(pinch[i]) and pinch[i] < 0.35:
            st[i] = 'PINCH'
        elif ext_med[i] < 1.05:
            st[i] = 'CLOSED'
        elif ext_med[i] > 1.40:
            st[i] = 'OPEN'
        else:
            st[i] = 'HALF_OPEN'
    # 15 fps pose estimates can flicker for one frame around a threshold.
    # Apply a small majority filter without filling invalid frames or changing
    # the underlying numeric evidence used by the interaction rules.
    raw = st.copy()
    for i in range(1, len(st) - 1):
        if raw[i] == 'UNKNOWN':
            continue
        win = [x for x in raw[i - 1:i + 2] if x != 'UNKNOWN']
        if len(win) >= 2:
            vals, counts = np.unique(win, return_counts=True)
            st[i] = vals[int(np.argmax(counts))]
    return st, ext_med


def ground_events(events, sd: Path):
    """用 grounding 证据仲裁每事件的物体（保守三态，不强制取最近）。

    规则 v1：排除画面面积 >25% 的“环境类”（laptop 等容器性大框）；
    通过条件 = 窗口内 dist 中位 <0.95 且 near 帧占比 >0.25；
    通过 → resolved；有候选但无一通过 → vlm_detector_mismatch（保留 VLM 名）；
    手附近无任何检出 → no_object_near_hand。键盘 + 弱抓握 → 键盘触碰语义。
    """
    gdir = sd / 'grounding'
    if not (gdir / 'association_evidence.parquet').exists():
        return events, None
    a = pd.read_parquet(gdir / 'association_evidence.parquet')
    tr = pd.read_parquet(gdir / 'object_tracks.parquet')
    area = (tr.x1 - tr.x0) * (tr.y1 - tr.y0)
    big = {c for c in COCO_BIG_AREA if
           (area[tr.cls == c].median() if (tr.cls == c).any() else 0) > 0.25}
    for e in events:
        w = a[(a.hand == e['hand_side'])
              & (a.frame_t >= e['start_time'] - 0.2)
              & (a.frame_t <= e['end_time'] + 0.2)
              & (~a.cls.isin(big))]
        ev = {}
        for cls, gg in w.groupby('cls'):
            ev[cls] = {'n': int(len(gg)),
                       'dist_median': round(float(gg.dist.median()), 3),
                       'iou_median': round(float(gg.iou.median()), 3),
                       # Only count evidence rows inside this event.  The old
                       # track-wide near_frames value was repeated on every row
                       # and was unreliable for short, occluded interactions.
                       'near_ratio': round(float((gg.dist < 1.0).mean()), 3),
                       'overlap_ratio': round(float((gg.iou > 0.05).mean()), 3),
                       'corr_median': (round(float(gg.move_corr.median()), 3)
                                       if gg.move_corr.notna().any() else None)}
        passed = [c for c, v in ev.items()
                  if v['dist_median'] < 0.95 and v['near_ratio'] > 0.25]
        # v2 握持遮挡线索：手与物体 IoU 高 = 物体可能在手中/被手遮（漏检主因）
        held = [c for c, v in ev.items() if v['iou_median'] > 0.25]
        e['grounding'] = {'per_class': ev,
                          'environment_classes_excluded': sorted(big)}
        vlm = e['object_category']
        if isinstance(vlm, str):
            vlm = vlm.strip().lower()
            vlm = {'手机': 'cell phone', 'phone': 'cell phone',
                   'mobile phone': 'cell phone', 'smartphone': 'cell phone',
                   'cellphone': 'cell phone', 'computer mouse': 'mouse',
                   'computer keyboard': 'keyboard'}.get(vlm, vlm)
            e['object_category'] = vlm
        if len(passed) == 1:
            e['object_detected'] = passed[0]
            e['object_resolution'] = 'resolved'
            e['object_category'] = passed[0]          # 检测器优先
            e['object_source'] = 'detector'
        elif held:
            e['object_detected'] = held[0]
            e['object_resolution'] = 'held_occlusion_clue'
            e['object_source'] = 'iou_clue'
        elif ev:
            e['object_detected'] = None
            e['object_resolution'] = 'vlm_detector_mismatch'
            e['object_vlm_claim'] = vlm               # v2: VLM 降级为候选
            e['object_candidate'] = vlm
            e['object_category'] = None
            e['object_source'] = 'vlm_candidate'
            e['conflicts'] = sorted(set(e['conflicts'] +
                                        ['object_vlm_detector_mismatch']))
            e['needs_review'] = True
        else:
            e['object_detected'] = None
            e['object_resolution'] = 'no_object_near_hand'
            e['object_vlm_claim'] = vlm
            e['object_candidate'] = vlm
            e['object_category'] = None
            e['object_source'] = 'vlm_candidate'
            e['conflicts'] = sorted(set(e['conflicts'] + ['no_object_near_hand']))
            e['needs_review'] = True
        if vlm and e.get('object_category') != vlm and 'object_vlm_claim' not in e:
            e['object_vlm_claim'] = vlm               # 检测改名时保留 VLM 原话
        # 键盘触碰语义：解析到键盘且 3D 抓握证据弱 → 不称 HOLD/GRASP
        det = e.get('object_detected')
        if det == 'keyboard' and e.get('grasp_evidence', {}).get(
                'low_pinch_frame_pct', 100) < 30:
            e['semantic_action'] = 'keyboard_touch_candidate'
            e['semantic_action_before_grounding'] = e.get(
                'semantic_action_before_grounding', e['semantic_action'])
    return events, big


def refine_interaction_events(events):
    """把物体候选 + 动作/原语组合成可解释的交互事件。

    ``semantic_action`` 仍保留原有候选语义；对手机额外写入
    ``interaction_action``，这样检测器漏检但 VLM 提供手机候选时也能显示
    “拿起手机（候选）”，而不是把物体信息静默丢掉。
    """
    for e in events:
        obj = e.get('object_detected') or e.get('object_category') \
            or e.get('object_vlm_claim') or e.get('object_candidate')
        if not isinstance(obj, str):
            continue
        obj = obj.strip().lower()
        obj = {'手机': 'cell phone', 'phone': 'cell phone',
               'mobile phone': 'cell phone', 'smartphone': 'cell phone',
               'cellphone': 'cell phone', 'computer mouse': 'mouse',
               'computer keyboard': 'keyboard'}.get(obj, obj)
        e['object_candidate'] = obj
        if obj not in PHONE_ALIASES:
            continue

        prims = set(e.get('motion_primitives') or ())
        va = e.get('visual_action')
        if va == 'release' or 'release' in prims:
            action = 'put_down_phone'
        elif va == 'lift' or 'lift' in prims:
            action = 'pick_up_phone'
        elif va in ('transport', 'hold') or {'transport', 'hold'} & prims:
            action = 'hold_phone'
        elif va == 'grasp' or {'pinch', 'power_grasp'} & prims:
            action = 'pick_up_phone'
        else:
            continue

        e['interaction_object'] = 'cell phone'
        e['interaction_action'] = action
        e['interaction_action_cn'] = INTERACTION_CN[action]
        e['semantic_action_before_interaction'] = e.get('semantic_action')
        e['semantic_action'] = action
        if e.get('object_source') == 'vlm_candidate':
            e['interaction_status'] = 'candidate'
            e['needs_review'] = True
            e['conflicts'] = sorted(set(e.get('conflicts', []) +
                                        ['phone_detector_missing']))
        else:
            e['interaction_status'] = 'grounded'
    return events


COCO_BIG_AREA = {'laptop', 'tv'}


def side_of_seg(seg_id: str) -> str:
    return {'L': 'left', 'R': 'right', 'B': 'both'}[seg_id[0]]


# ---------------------------------------------------------------- 融合规则

def fuse_semantics(va: str, contact: str, obj, prims: list[str]) -> tuple[str, list[str]]:
    """返回 (semantic_action, conflicts)。原语仅辅助证据。"""
    c = []
    grasp_m = any(x in prims for x in ('pinch', 'power_grasp'))
    if va == 'grasp':
        sem = 'candidate_object_grasp'
        if not grasp_m:
            c.append('primitive_visual_mismatch')
    elif va == 'transport':
        sem = 'candidate_object_transport'
        if 'transport' not in prims:
            c.append('primitive_visual_mismatch')
    elif va == 'hold':
        sem = 'candidate_object_hold'
        if not ({'hold', 'transport', 'pinch', 'power_grasp'} & set(prims)):
            c.append('weak_hold_evidence')
    elif va == 'rotate':
        sem = 'candidate_object_rotate' if obj else 'hand_rotation_no_object_evidence'
        if 'rotate' not in prims:
            c.append('primitive_visual_mismatch')
    elif va == 'release':
        sem = 'candidate_release'
    elif va == 'approach':
        sem = 'candidate_approach'
    elif va == 'lift':
        sem = 'candidate_lift'
    elif va == 'gesture':
        sem = 'gesture' if obj is None else 'gesture_with_object_nearby'
    elif va == 'unknown':
        sem = 'unknown_action'
    else:                       # 词表外（如 'type'）：保留原词，进复核
        sem, _ = 'needs_semantic_review', c.append('visual_action_out_of_vocab')
    if va in ('grasp', 'lift', 'transport') and contact == 'no_contact':
        c.append('contact_action_inconsistent')
    return sem, c


# ---------------------------------------------------------------- 主流程

def build(sd: Path, out_dir: Path, decisions: list | None = None):
    demo = pd.read_parquet(sd / 'demonstration.parquet')
    segs = json.loads((sd / 'action_segments.json').read_text())
    clips = {c['clip_id']: c for c in json.loads(
        (sd / 'visual_context/clip_manifest.json').read_text())}
    vp = json.loads((sd / 'visual_semantics/visual_predictions.json').read_text())
    ts = demo['timestamp'].to_numpy(dtype=float)
    fi = demo['frame_index'].to_numpy(dtype=int)
    inputs_meta = {str(sd / n): md5(sd / n) for n in (
        'demonstration.parquet', 'action_segments.json',
        'visual_context/clip_manifest.json',
        'visual_semantics/visual_predictions.json')}

    def frames_between(lo, hi):
        i0 = int(np.searchsorted(ts, lo - 1e-6))
        i1 = int(np.searchsorted(ts, hi + 1e-6, 'right')) - 1
        return max(0, i0), min(len(ts) - 1, max(i0, i1))

    seg3d = {'left': segs['hands']['left'], 'right': segs['hands']['right']}

    events, stats = [], {'duplicates_corroborating': 0, 'merged': 0,
                         'possible_duplicates': 0}
    consumed = [p for p in vp['predictions'] if 'duplicate_of' not in p]
    stats['duplicates_corroborating'] = len(vp['predictions']) - len(consumed)
    stats['predictions_consumed'] = len(vp['predictions'])   # 全部并入分组
    # 分组：同 clip × 同手 × 同视觉动作 = 同一交互。带 duplicate_of 的对目
    # 预测也要并入——那是 Step 2.2 的跨目去重标记，不是丢弃证据。
    groups = {}
    for p in vp['predictions']:
        hands = ('left', 'right') if p['hand_side'] == 'both' else (p['hand_side'], )
        for hand in hands:
            groups.setdefault((p['clip_id'], hand, p['visual_action']), []).append(p)

    for (cid, hand, va), plist in sorted(groups.items()):
        plist.sort(key=lambda p: p['camera'] == 'R')      # camera0 证据为主
        p = plist[0]
        clip = clips[cid]
        # 时间：视觉估计(相对clip) → 窗口时间；组内取并集；与 3D 段两套并存
        wins = []
        for x in plist:
            te = x.get('vlm_time_estimate') or {}
            lo_ = cid and clip['start_time'] + (te.get('start_s') or 0.0)
            hi_ = clip['start_time'] + (te.get('end_s')
                                        if te.get('end_s') is not None
                                        else (clip['end_time'] - clip['start_time']))
            wins.append((max(lo_, clip['start_time']),
                         min(hi_, clip['end_time'])))
        v_lo, v_hi = max(min(w[0] for w in wins), clip['start_time']), \
            min(max(w[1] for w in wins), clip['end_time'])
        # 3D 段：clip 段 id 中属于本手的（L#/R#/B# → 对应手）
        my_segs = [s for s in clip['source_segment_ids']
                   if side_of_seg(s) in (hand, 'both')]
        m3 = [dict(seg, seg_id=sid) for sid in my_segs
              for seg in seg3d[hand]
              if seg3d[hand].index(seg) == int(sid[1:])]
        m_lo = min((s['start_ts'] for s in m3), default=v_lo)
        m_hi = max((s['end_ts'] for s in m3), default=v_hi)
        conflicts = []
        if abs(m_lo - v_lo) > BOUNDARY_DIFF_S or abs(m_hi - v_hi) > BOUNDARY_DIFF_S:
            conflicts.append('time_boundary_mismatch')
        lo, hi = max(v_lo, m_lo) if m3 else v_lo, min(v_hi, m_hi) if m3 else v_hi
        if hi <= lo:
            lo, hi = v_lo, v_hi            # 无交集：用视觉窗并记冲突
            conflicts.append('no_temporal_overlap')

        prims = sorted({s['primitive'] for s in m3} or
                       set(clip[f'{hand}_primitive_candidates']))
        sem, cfl = fuse_semantics(p['visual_action'], p['contact_state'],
                                  p['object_category'], prims)
        conflicts += cfl
        if p.get('cross_camera') == 'needs_review':
            conflicts.append('camera_disagreement')

        f0, f1 = frames_between(lo, hi)

        # release 硬截断：事件内最后一个低 pinch 帧 +2 帧缓冲为持握终点；
        # 抓握证据强度（pinch 是辅助证据，不能证明抓取）
        pin_ev = demo[f'{hand}_pinch_idx'].iloc[f0:f1 + 1].to_numpy(float)
        low = np.isfinite(pin_ev) & (pin_ev < 0.35)
        grasp_ev = {'low_pinch_frame_pct': round(float(low.mean()) * 100, 0),
                    'pinch_median': round(float(np.nanmedian(pin_ev)), 3)}
        if grasp_ev['low_pinch_frame_pct'] < 30:
            conflicts.append('weak_grasp_evidence')
        end_refined, _end_orig = None, hi
        if low.any():
            cut_ts = float(ts[f0 + int(np.where(low)[0][-1])]) + 2 / 15.0
            if cut_ts < hi - 0.3:
                end_refined = round(cut_ts, 3)
                conflicts.append('boundary_refined_by_release')
                hi = end_refined
                f0, f1 = frames_between(lo, hi)
        sts = demo[f'{hand}_status'].iloc[f0:f1 + 1]


        q = {k: round(float((sts == k).mean()), 3)
             for k in ('observed', 'interpolated', 'invalid') if (sts == k).any()}
        invalid_pct = float((sts == 'invalid').mean()) if len(sts) else 0.0
        if invalid_pct > INVALID_REVIEW_PCT:
            conflicts.append('low_quality_invalid_frames')

        obj = p['object_category']
        names = {x.get('object_category') for x in plist}
        if len(names) > 1:
            conflicts.append('object_naming_mismatch')
        obj_id = f"{cid}::{(obj or 'unknown').replace(' ', '_').lower()}" \
            if obj else None

        cameras = sorted({x['camera'] for x in plist})

        events.append({
            'event_id': None, 'source_clip_ids': [cid],
            'source_segment_ids': my_segs,
            'source_prediction_ids': [x['prediction_id'] for x in plist],
            'start_time': round(float(lo), 3), 'end_time': round(float(hi), 3),
            'visual_time_boundaries': [round(v_lo, 3), round(v_hi, 3)],
            'motion_time_boundaries': ([round(m_lo, 3), round(m_hi, 3)]
                                       if m3 else None),
            'start_frame': int(fi[f0]), 'end_frame': int(fi[f1]),
            'trajectory_rows': [f0, f1],
            'hand_side': hand,
            'object_category': obj,
            'object_category_alt': sorted(str(n) for n in names - {obj}),
            'object_instance_local_id': obj_id,
            'visual_action': p['visual_action'],
            'motion_primitives': prims,
            'semantic_action': sem,
            'contact_state': p['contact_state'],
            'motion_contact_cue': grasp_ev['pinch_median'],
            'grasp_evidence': grasp_ev,
            'end_time_original': (round(float(_end_orig), 3)
                                  if end_refined is not None else None),
            'trajectory_reference': {
                'parquet': 'demonstration.parquet',
                'rows': [f0, f1],
                'timestamp_range': [round(float(ts[f0]), 3), round(float(ts[f1]), 3)]},
            'evidence_frames': {
                'video': p['evidence_frame_indices'],
                'video_timestamps': p['evidence_timestamps'],
                'trajectory_rows': list(range(f0, f1 + 1))},
            'camera_evidence': sorted(cameras),
            'alignment_status': clip['alignment_status'],
            'alignment_uncertainty': TIME_UNC,
            'quality_status': q,
            'conflicts': sorted(set(conflicts)),
            'needs_review': bool(conflicts),
            'verification_status': 'pending_review'})

    # ---- grounding 物体仲裁（Step 2.5）
    events, _big = ground_events(events, sd)
    # ---- 交互事件语义：物体候选 + 手势/运动原语 + 释放时序
    events = refine_interaction_events(events)

    # ---- 跨片段物体连续性：同手相邻事件物体跳变（<5s 间隔）→ review
    for hand in ('left', 'right'):
        chain = sorted([e for e in events if e['hand_side'] == hand
                        and e['object_category']],
                       key=lambda e: e['start_time'])
        for a, b in zip(chain, chain[1:]):
            if (b['start_time'] - a['end_time']) < 5.0 \
                    and a['object_category'] != b['object_category']:
                names = {a['object_category'], b['object_category']}
                b['conflicts'] = sorted(set(b['conflicts'] +
                                            ['object_continuity_question']))
                b.setdefault('object_continuity_alt', sorted(str(n) for n in names))
                b['needs_review'] = True

    # ---- 双手同物体标记（同 clip 内时间重叠 + 同局部物体 id）
    for i, e in enumerate(events):
        for e2 in events[i + 1:]:
            if (e['object_instance_local_id'] and
                    e['object_instance_local_id'] == e2['object_instance_local_id'] and
                    e['hand_side'] != e2['hand_side'] and
                    min(e['end_time'], e2['end_time']) > max(e['start_time'], e2['start_time'])):
                e['bimanual_same_object'] = e2['bimanual_same_object'] = True

    # ---- 跨 clip 去重（同手+同物体+同语义+时间重叠）
    events.sort(key=lambda e: (e['start_time'], e['hand_side']))
    for i, e in enumerate(events):
        if 'merged_into' in e:
            continue
        for e2 in events[i + 1:]:
            if 'merged_into' in e2 or e2['hand_side'] != e['hand_side']:
                continue
            same = (e2['object_instance_local_id'] == e['object_instance_local_id']
                    and e2['semantic_action'] == e['semantic_action'])
            ov = min(e['end_time'], e2['end_time']) - max(e['start_time'], e2['start_time'])
            if same and ov > 0:
                shorter = min(e['end_time'] - e['start_time'], e2['end_time'] - e2['start_time'])
                if ov >= 0.6 * shorter:            # 高重叠 → 合并（保留两套证据）
                    e['end_time'] = max(e['end_time'], e2['end_time'])
                    e['end_frame'] = max(e['end_frame'], e2['end_frame'])
                    e['trajectory_rows'][1] = max(e['trajectory_rows'][1], e2['trajectory_rows'][1])
                    e['source_clip_ids'] = sorted(set(e['source_clip_ids'] + e2['source_clip_ids']))
                    e['source_prediction_ids'] += e2['source_prediction_ids']
                    e['camera_evidence'] = sorted(set(e['camera_evidence'] + e2['camera_evidence']))
                    e2['_merged_flag'] = True
                    stats['merged'] += 1
                else:                               # 低重叠 → 疑似重复，各自保留
                    e2.setdefault('possible_duplicate_of', []).append(e)
                    stats['possible_duplicates'] += 1
    kept = [e for e in events if '_merged_flag' not in e]
    for k, e in enumerate(kept):
        e['event_id'] = f'hoi_{k:04d}'
        e['possible_duplicate_of'] = [x['event_id'] for x in e.get('possible_duplicate_of', [])]

    # ---- 人工裁决应用（Step 2.6）：在逐帧标签生成前生效，保证三产物同源
    repair_log = []
    if decisions:
        for dec in decisions:
            e = next((x for x in kept if x['event_id'] == dec['event_id']), None)
            if e is None:
                repair_log.append({'event_id': dec['event_id'], 'status': 'skipped_no_event'})
                continue
            lab = dec['reviewed_label']
            t0, t1 = lab['bounds']
            if not (0 <= t0 < t1 <= float(ts[-1]) + 1e-6):
                repair_log.append({'event_id': dec['event_id'], 'status': 'skipped_bad_bounds'})
                continue
            e['human_decision'] = {
                'action': dec['action'], 'reviewer': dec['reviewer'],
                'reason': dec['correction_reason'], 'review_time': dec['review_time'],
                'provenance': dec.get('provenance', 'human_review'),
                'original': dec['original_prediction']}
            if dec['action'] == 'mark_unknown':
                e['object_category'] = None
                e['semantic_action'] = 'unknown_action'
            elif dec['action'] == 'mark_no_interaction':
                e['object_category'] = None
                e['semantic_action'] = 'no_interaction'
            else:
                e['hand_side'] = lab['hand']
                e['object_category'] = lab['object']
                e['semantic_action'] = lab['semantic']
            e['start_time'], e['end_time'] = round(t0, 3), round(t1, 3)
            e['start_frame'] = int(fi[frames_between(t0, t1)[0]])
            e['end_frame'] = int(fi[frames_between(t0, t1)[1]])
            e['trajectory_rows'] = list(frames_between(t0, t1))
            e['needs_review'] = dec['action'] == 'keep_reviewing'
            e['verification_status'] = ('pending_review' if e['needs_review']
                                        else 'human_reviewed')
            e['conflicts'] = [] if dec['action'] == 'confirm' else e['conflicts']
            repair_log.append({'event_id': dec['event_id'], 'status': 'applied',
                               'action': dec['action'], 'reviewer': dec['reviewer']})

    # ---- 逐帧标签（姿态通道 + 交互状态机 + 展示标签，供渲染与审计同源使用）
    poses = {h: pose_state_series(demo, h)[0] for h in ('left', 'right')}
    rows = {'frame_index': fi.tolist(), 'timestamp': ts.tolist()}
    for hand in ('left', 'right'):
        cols = {k: [] for k in ('event_ids', 'semantic_action', 'object_category',
                                'contact_state', 'verification_status', 'needs_review')}
        for r in range(len(ts)):
            act = [e for e in kept if e['hand_side'] == hand
                   and e['trajectory_rows'][0] <= r <= e['trajectory_rows'][1]]
            main = sorted(act, key=lambda e: -e['needs_review'])[:1]  # 主事件优先无冲突者
            cols['event_ids'].append(','.join(e['event_id'] for e in act))
            cols['semantic_action'].append(main[0]['semantic_action'] if main else '')
            cols['object_category'].append(str(main[0]['object_category']) if main else '')
            cols['contact_state'].append(main[0]['contact_state'] if main else '')
            cols['verification_status'].append(main[0]['verification_status'] if main else '')
            cols['needs_review'].append(bool(main and main[0]['needs_review']))
        rows[f'{hand}_event_ids'] = cols['event_ids']
        rows.update({f'{hand}_{k}': v for k, v in cols.items() if k != 'event_ids'})
        rows[f'{hand}_status'] = demo[f'{hand}_status'].tolist()

    # 交互状态机逐帧 + 展示标签（渲染器直接消费 display 列，保证一致）
    for hand in ('left', 'right'):
        inter = np.array(['NO_INTERACTION'] * len(ts), dtype=object)
        for e in kept:
            if e['hand_side'] != hand:
                continue
            r0, r1 = e['trajectory_rows']
            r1 = min(r1 + 1, len(ts) - 1)
            object_evidence = (e.get('object_detected') or e.get('object_category')
                               or e.get('object_vlm_claim') or e.get('object_candidate'))
            contact_evidence = e.get('contact_state') in ('visible_contact',
                                                            'probable_contact')
            for r in range(r0, r1 + 1):
                pin = demo[f'{hand}_pinch_idx'].iloc[r]
                pose = poses[hand][r]
                if (pose == 'PINCH' and np.isfinite(pin) and pin < 0.35
                        and (contact_evidence or object_evidence)):
                    inter[r] = 'GRASP'
                elif pose == 'PINCH' and np.isfinite(pin) and pin < 0.35:
                    # A pinch is a gesture by itself; do not call it a grasp
                    # when no object/contact evidence exists.
                    inter[r] = 'NO_INTERACTION'
                elif r >= r1 - 2 and pose == 'OPEN':
                    inter[r] = 'RELEASE'
                else:
                    inter[r] = 'HOLD'
            for r in range(max(0, r0 - 8), r0):        # 事件前 0.5s 高速接近
                sp = demo[f'{hand}_trans_speed'].iloc[r]
                if np.isfinite(sp) and sp > 0.30:
                    inter[r] = 'APPROACH'
        # 手机持握延续链：拿起→持握(延续至张开/放下)→放下(数帧)
        held = [''] * len(ts)
        pu = [e for e in kept if e['hand_side'] == hand
              and e['semantic_action'] == 'pick_up_phone']
        pdn = [e for e in kept if e['hand_side'] == hand
               and e['semantic_action'] == 'put_down_phone']
        for e in pu:
            r0, r1 = e['trajectory_rows']
            for r in range(r0, min(r1 + 1, len(ts))):
                held[r] = '拿起手机'
            r = r1 + 1
            while r < len(ts) and r - r1 <= 150:
                if any(a['trajectory_rows'][0] <= r <= a['trajectory_rows'][1]
                       for a in pdn):
                    break
                # 握手机时手型常为张开级伸展，pose 不可作为放下判据；
                # 持握延续仅由 put_down 事件或时间上限终止
                held[r] = '持握手机'
                r += 1
        for e in pdn:
            for r in range(e['trajectory_rows'][0],
                           min(e['trajectory_rows'][1] + 1, len(ts))):
                held[r] = '放下手机'
        disp = []
        for r in range(len(ts)):
            p, i = poses[hand][r], inter[r]
            obj = ''
            interaction = None
            for e in kept:
                if e['hand_side'] == hand and e['trajectory_rows'][0] <= r <= e['trajectory_rows'][1]:
                    obj = str(e.get('object_detected') or e.get('object_category')
                              or e.get('object_vlm_claim') or e.get('object_candidate') or '')
                    interaction = e.get('interaction_action')
                    break
            obj = OBJECT_CN.get(obj.lower(), obj)
            obj_ok = obj not in ('', 'None', 'nan', 'unknown_action', '无交互')
            if held[r]:                      # 手机持握延续链优先（拿起/持握/放下）
                core = held[r] + ' | 手机'
            elif interaction:
                core = INTERACTION_CN[interaction]
                if any(e.get('interaction_action') == interaction
                       and e.get('interaction_status') == 'candidate'
                       and e['hand_side'] == hand
                       and e['trajectory_rows'][0] <= r <= e['trajectory_rows'][1]
                       for e in kept):
                    core += '（候选）'
            else:
                core = f'{INTER_CN[i]}' + (f' | {obj}' if obj and obj_ok
                                           and i in ('GRASP', 'HOLD') else '')
            disp.append(f'{POSE_CN[p]} | {core}')
        rows[f'{hand}_pose_state'] = poses[hand]
        # Explicit gesture channel: kept separate from interaction_state so a
        # pinch without an object remains a gesture rather than a false grasp.
        rows[f'{hand}_gesture_state'] = poses[hand]
        rows[f'{hand}_interaction_state'] = inter
        rows[f'{hand}_display'] = disp

    out_dir.mkdir(parents=True, exist_ok=True)
    ann = {'meta': {'schema_version': SCHEMA_VERSION,
                    'fusion_rules_version': FUSION_RULES_VERSION,
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'inputs_md5': inputs_meta,
                    'time_uncertainty': TIME_UNC,
                    'notes': 'multimodal fusion CANDIDATES; primitives are '
                             'auxiliary evidence only; not robot ground truth'},
           'events': kept}
    (out_dir / 'hoi_annotations.json').write_text(
        json.dumps(ann, ensure_ascii=False, indent=2))
    pd.DataFrame(rows).to_parquet(out_dir / 'hoi_frame_labels.parquet', index=False)

    confl = {}
    for e in kept:
        for c in e['conflicts']:
            confl[c] = confl.get(c, 0) + 1
    review = [{'event_id': e['event_id'], 'reasons': e['conflicts'] or ['routine_check'],
               'evidence': {'keyframes_dir': 'visual_semantics/review',
                            'trajectory_rows': e['trajectory_rows'],
                            'video_frames': e['evidence_frames']['video']}}
              for e in kept if e['needs_review']]
    (out_dir / 'repair_log.json').write_text(
        json.dumps(repair_log, ensure_ascii=False, indent=2))
    (out_dir / 'review_queue.json').write_text(
        json.dumps(review, ensure_ascii=False, indent=2))
    (out_dir / 'fusion_report.json').write_text(json.dumps({
        'predictions_total': len(vp['predictions']),
        **stats,
        'events': len(kept),
        'unknown_object_events': sum(1 for e in kept if not e['object_category']),
        'bimanual_same_object_events': sum(1 for e in kept if e.get('bimanual_same_object')),
        'conflicts_by_type': confl,
        'review_queue': len(review)}, ensure_ascii=False, indent=2))
    return ann, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--decisions', default=None,
                    help='review_decisions.jsonl：人工裁决应用到事件（输出到 hoi_repaired，不覆盖原结果）')
    args = ap.parse_args()
    sd = Path(args.structured_dir)
    out = Path(args.out_dir) if args.out_dir else sd / 'hoi_fusion'
    decisions = None
    if args.decisions:
        decisions = [json.loads(l) for l in
                     Path(args.decisions).read_text().splitlines() if l.strip()]
        out = out if args.out_dir else sd / 'hoi_repaired'
    ann, stats = build(sd, out, decisions)
    evs = ann['events']
    print(f'[fuse_hoi] 事件 {len(evs)} 个（合并 {stats["merged"]}，疑似重复 '
          f'{stats["possible_duplicates"]}）-> {out}')
    for e in evs:
        print(f"  {e['event_id']} {e['start_time']:6.2f}-{e['end_time']:6.2f}s "
              f"{e['hand_side']:5} obj={str(e['object_category']):16} "
              f"{e['semantic_action']:32} prims={e['motion_primitives']} "
              f"cam={''.join(e['camera_evidence'])} "
              f"{'REVIEW:' + ','.join(e['conflicts']) if e['conflicts'] else ''}")


if __name__ == '__main__':
    main()
