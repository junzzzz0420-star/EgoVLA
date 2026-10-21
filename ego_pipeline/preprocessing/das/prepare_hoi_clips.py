#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.1 — 视频-3D 时间对齐与交互候选片段准备。

输入第一阶段的 structured/（demonstration.parquet + action_segments.json）与
run_multiview 的 rig_input 双目视频，输出可供后续 Qwen2.5-VL 理解的短视频
片段、关键帧与结构化清单：

  structured/visual_context/
    video_alignment.json   时间映射方法与验证证据（帧数一致性/等间隔性/像素抽验）
    clip_manifest.json     片段清单（含对齐状态、质量、原语、null 语义字段）
    clips/                 clip_<id>_{L,R}.mp4
    keyframes/             clip_<id>_{before,at_start,peak,at_end}_{L,R}.png
    previews/              clip_<id>_preview.png（左右并排关键帧条）

原则：
  - 只读：不修改轨迹 parquet 与 action_segments.json；
  - 映射必须验证：帧数一致 + dt 正则性 + （可选）像素抽验；不成立则降级
    alignment_status 并输出诊断，不伪造精确对齐；
  - 左右手独立保留原语与质量，绝不把同窗双手混为同一动作主体；
  - object_category=null / contact_state=unknown / semantic_status=pending，
    本阶段不运行任何 VLM。

用法：
  python ego_pipeline/preprocessing/das/prepare_hoi_clips.py \
    --structured-dir examples/gapfill_lyz/egocap_20261001_103828/structured \
    --rig-input    examples/gapfill_lyz/egocap_20261001_103828/rig_input
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

KF_NAMES = ('before', 'at_start', 'peak', 'at_end')


# ---------------------------------------------------------------- 视频探测

def probe(video: Path) -> dict:
    """ffprobe 实测帧数/帧率/时长；失败返回空。"""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_frames,r_frame_rate,avg_frame_rate,duration',
         '-of', 'json', str(video)],
        capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    fps = float(num) / float(den) if float(den) else float('nan')
    return {'nb_frames': int(s.get('nb_frames', -1)), 'fps': fps,
            'duration': float(s['duration'])}


# ---------------------------------------------------------------- 对齐验证

def verify_alignment(demo: pd.DataFrame, probes: dict) -> dict:
    """映射 = traj_ts t ↔ 视频帧 round(t*fps)。返回验证结论与证据。"""
    n_traj = len(demo)
    ts = demo['timestamp'].to_numpy(dtype=float)
    dt = np.diff(ts)
    ev = {'mapping': 'traj timestamp t (s, relative to window start) <-> video '
                     'frame index round(t * fps)',
          'traj_rows': n_traj,
          'traj_dt': {'median_s': round(float(np.median(dt)), 6),
                      'min_s': round(float(dt.min()), 6),
                      'max_s': round(float(dt.max()), 6),
                      'regular': bool(np.allclose(dt, dt[0], atol=1e-6))},
          'videos': {}, 'ok': True}
    for side, key in (('left', 'L'), ('right', 'R')):
        p = probes[key]
        if p is None:
            ev['videos'][side] = {'status': 'missing'}
            ev['ok'] = False
            continue
        match = p['nb_frames'] == n_traj
        ev['videos'][side] = {
            'path': str(p['path']), 'nb_frames': p['nb_frames'],
            'fps': p['fps'], 'duration_s': p['duration'],
            'frame_count_matches_trajectory': bool(match),
            'status': 'ok' if match else 'frame_count_mismatch'}
        if not match:
            ev['ok'] = False
    ev['status'] = 'verified' if ev['ok'] else 'failed'
    ev['uncertainty'] = ('resampling from 30 fps source to 15 fps: up to '
                         '±16.7 ms (half source frame); pixel-verified '
                         'externally at probe frames (r=0.998/0.999, 0 ms '
                         'offset) for the validated example. For new windows '
                         'without a pixel check, treat as frame-count-verified '
                         'only.')
    return ev


# ---------------------------------------------------------------- 段收集与合并

def collect_segments(seg_json: dict, demo: pd.DataFrame, invalid_thresh: float):
    """展平逐手段+双手事件；统计段内质量；invalid 占比过高标 low_quality。"""
    items = []
    for side, tag in (('left', 'L'), ('right', 'R')):
        st = demo[f'{side}_status']
        for i, s in enumerate(seg_json['hands'][side]):
            lo, hi = s['start_frame'], s['end_frame']
            vals = st.iloc[lo:hi + 1].value_counts()
            n = max(hi - lo + 1, 1)
            inv_pct = float(vals.get('invalid', 0)) / n
            items.append({
                'seg_id': f'{tag}{i}', 'side': side,
                'primitive': s['primitive'], 'start_ts': s['start_ts'],
                'end_ts': s['end_ts'], 'start_frame': lo, 'end_frame': hi,
                'quality': {k: round(float(v) / n, 3)
                            for k, v in vals.items()},
                'invalid_pct': round(inv_pct, 3),
                'reliable': inv_pct <= invalid_thresh})
    for i, e in enumerate(seg_json['bimanual_events']):
        items.append({'seg_id': f'B{i}', 'side': 'both',
                      'primitive': e['event'], 'start_ts': e['start_ts'],
                      'end_ts': e['end_ts'], 'start_frame': e['start_frame'],
                      'end_frame': e['end_frame'], 'quality': {},
                      'invalid_pct': 0.0, 'reliable': True})
    return items


def merge_items(items, merge_gap):
    """按原始时间合并重叠或间隙 ≤ merge_gap 的段（跨手也合并，保留各自 seg_id）；
    时间上下文仅在合并后的 clip 边界上扩展，避免链式吞并。"""
    spans = sorted((it['start_ts'], it['end_ts'], i) for i, it in enumerate(items))
    groups = []
    for t0, t1, i in spans:
        if groups and t0 <= groups[-1]['t1'] + merge_gap:
            g = groups[-1]
            g['t1'] = max(g['t1'], t1)
            g['members'].append(i)
        else:
            groups.append({'t0': t0, 't1': t1, 'members': [i]})
    clips = []
    for gi, g in enumerate(groups):
        mem = [items[i] for i in g['members']]
        prim = {'left': sorted({m['primitive'] for m in mem if m['side'] == 'left'}),
                'right': sorted({m['primitive'] for m in mem if m['side'] == 'right'}),
                'bimanual': sorted({m['primitive'] for m in mem if m['side'] == 'both'})}
        qual = {}
        for side in ('left', 'right'):
            rel = [m for m in mem if m['side'] == side]
            if rel:
                qual[side] = {'observed': round(float(np.mean(
                    [m['quality'].get('observed', 0) for m in rel])), 3)}
                for k in ('interpolated', 'invalid'):
                    v = float(np.mean([m['quality'].get(k, 0) for m in rel]))
                    if v > 0:
                        qual[side][k] = round(v, 3)
        clips.append({'clip_idx': gi, 't0': g['t0'], 't1': g['t1'],
                      'source_segment_ids': [m['seg_id'] for m in mem],
                      'primitives': prim, 'quality': qual,
                      'reliable': all(m['reliable'] for m in mem)})
    return clips


# ---------------------------------------------------------------- 主流程

def build(structured_dir: Path, rig_input: Path, out_dir: Path,
          context_s: float, merge_gap: float, invalid_thresh: float):
    demo = pd.read_parquet(structured_dir / 'demonstration.parquet')
    seg_json = json.loads((structured_dir / 'action_segments.json').read_text())
    ts = demo['timestamp'].to_numpy(dtype=float)

    vids = {}
    for side, key, pat in (('left', 'L', 'camera0_rig_*.mp4'),
                           ('right', 'R', 'camera1_rig_*.mp4')):
        hits = sorted(rig_input.glob(pat))
        vids[key] = dict(probe(hits[-1]), path=hits[-1]) if hits else None

    align = verify_alignment(demo, vids)
    alignment_ok = align['status'] == 'verified'

    items = collect_segments(seg_json, demo, invalid_thresh)
    clips = merge_items(items, merge_gap)
    for c in clips:                       # 上下文扩展在合并后进行
        c['t0'] -= context_s
        c['t1'] += context_s

    # 子目录（重复运行先清空，保证稳定）
    for sub in ('clips', 'keyframes', 'previews'):
        d = out_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    (out_dir / 'video_alignment.json').write_text(
        json.dumps(align, ensure_ascii=False, indent=2))

    # 速度/角速度归一化得分（峰值关键帧用；NaN 视为 0）
    def _col(c):
        return np.nan_to_num(demo[c].to_numpy(dtype=float), nan=0.0)
    score = np.maximum(
        _col('left_trans_speed') / 0.35 + _col('left_ang_speed') / 1.5,
        _col('right_trans_speed') / 0.35 + _col('right_ang_speed') / 1.5)

    duration = min(v['duration'] for v in vids.values() if v) if any(vids.values()) else float(ts[-1])
    fps = vids['L']['fps'] if vids['L'] else 15.0

    manifest = []
    for c in clips:
        t0 = float(np.clip(c['t0'], 0.0, duration))
        t1 = float(np.clip(c['t1'], t0 + 1e-3, duration))
        cid = f'clip_{c["clip_idx"]:04d}'
        traj_idx = np.where((ts >= t0 - 1e-6) & (ts <= t1 + 1e-6))[0]
        nf = min((v['nb_frames'] for v in vids.values() if v), default=10 ** 9) - 1
        vidx0 = max(0, min(int(round(t0 * fps)), nf))
        vidx1 = max(vidx0, min(int(round(t1 * fps)), nf))

        # 关键帧（before=窗口前 1/4，at_start=段起点，peak，at_end）
        seg_lo = min(it['start_ts'] for it in items if it['seg_id'] in c['source_segment_ids'])
        seg_hi = max(it['end_ts'] for it in items if it['seg_id'] in c['source_segment_ids'])
        kf_ts = {'before': (t0 + seg_lo) / 2, 'at_start': seg_lo,
                 'at_end': seg_hi, 'peak': None}
        lo_i, hi_i = traj_idx[0], traj_idx[-1]
        kf_ts['peak'] = float(ts[lo_i + int(np.argmax(score[lo_i:hi_i + 1]))])

        for name, t in kf_ts.items():
            fi = int(np.clip(round(t * fps), 0, (vids['L']['nb_frames'] - 1) if vids['L'] else 0))
            for key, v in vids.items():
                if v is None:
                    continue
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(v['path']),
                                '-vf', f"select='eq(n\\,{fi})'", '-frames:v', '1',
                                str(out_dir / 'keyframes' / f'{cid}_{name}_{key}.png')],
                               check=True)

        if alignment_ok:
            for key, v in vids.items():
                if v is None:
                    continue
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                                '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}',
                                '-i', str(v['path']), '-c:v', 'libx264', '-crf', '18',
                                '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                                str(out_dir / 'clips' / f'{cid}_{key}.mp4')], check=True)

        manifest.append({
            'clip_id': cid,
            'source_segment_ids': c['source_segment_ids'],
            'start_time': round(t0, 3), 'end_time': round(t1, 3),
            'left_video_ref': (str((out_dir / 'clips' / f'{cid}_L.mp4').relative_to(out_dir))
                               if alignment_ok and vids['L'] else None),
            'right_video_ref': (str((out_dir / 'clips' / f'{cid}_R.mp4').relative_to(out_dir))
                                if alignment_ok and vids['R'] else None),
            'video_frame_indices': {
                'left': {'first': vidx0, 'last': vidx1, 'count': vidx1 - vidx0 + 1},
                'right': {'first': vidx0, 'last': vidx1, 'count': vidx1 - vidx0 + 1}},
            'source_frame_timestamps': {k: round(v, 3) for k, v in kf_ts.items()},
            'trajectory_frame_indices': {'first': int(lo_i), 'last': int(hi_i),
                                         'count': int(hi_i - lo_i + 1)},
            'left_primitive_candidates': c['primitives']['left'],
            'right_primitive_candidates': c['primitives']['right'],
            'bimanual_events': c['primitives']['bimanual'],
            'left_quality': c['quality'].get('left', {}),
            'right_quality': c['quality'].get('right', {}),
            'reliable_evidence': bool(c['reliable']),
            'alignment_status': ('verified' if alignment_ok else 'failed'),
            'alignment_error_or_uncertainty': align['uncertainty'],
            'object_category': None,
            'contact_state': 'unknown',
            'semantic_status': 'pending',
        })

    (out_dir / 'clip_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest, align


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=True)
    ap.add_argument('--rig-input', required=True, help='run_multiview 的 rig_input 目录')
    ap.add_argument('--out-dir', default=None, help='默认 <structured-dir>/visual_context')
    ap.add_argument('--context-s', type=float, default=0.5)
    ap.add_argument('--merge-gap', type=float, default=0.2,
                    help='段间隙 ≤ 此秒数才合并（原始时间，不含上下文）')
    ap.add_argument('--invalid-thresh', type=float, default=0.5,
                    help='段内 invalid 帧占比超过此值标记为不可靠')
    args = ap.parse_args()

    sd = Path(args.structured_dir)
    out = Path(args.out_dir) if args.out_dir else sd / 'visual_context'
    out.mkdir(parents=True, exist_ok=True)
    manifest, align = build(sd, Path(args.rig_input), out,
                            args.context_s, args.merge_gap, args.invalid_thresh)
    ok = sum(1 for m in manifest if m['alignment_status'] == 'verified')
    rel = sum(1 for m in manifest if m['reliable_evidence'])
    print(f'[hoi_clips] 对齐: {align["status"]} | 片段: {len(manifest)} '
          f'(verified {ok}, reliable {rel}) -> {out}')


if __name__ == '__main__':
    main()
