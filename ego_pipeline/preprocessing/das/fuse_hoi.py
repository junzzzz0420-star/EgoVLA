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
FUSION_RULES_VERSION = 'v1'
TIME_UNC = ('+-16.7 ms video resampling (Step 2.1 verified mapping); '
            'vlm_time_estimate itself is a coarse model guess')
BOUNDARY_DIFF_S = 0.3     # 视觉/3D 边界差异超过此值记冲突
INVALID_REVIEW_PCT = 0.2  # 事件帧 invalid 占比超过此值 → low_quality


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


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

def build(sd: Path, out_dir: Path):
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
            'motion_contact_cue': round(float(np.nanmedian(
                demo[f'{hand}_pinch_idx'].iloc[f0:f1 + 1])), 3)
                if len(sts) else None,
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

    # ---- 逐帧标签
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
    args = ap.parse_args()
    sd = Path(args.structured_dir)
    out = Path(args.out_dir) if args.out_dir else sd / 'hoi_fusion'
    ann, stats = build(sd, out)
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
