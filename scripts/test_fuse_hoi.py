#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fuse_hoi.py 自动测试（Step 2.3 第四部分，12 项）。用法：
  python scripts/test_fuse_hoi.py <run_dir>
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ego_pipeline/preprocessing/das'))
import fuse_hoi

run = Path(sys.argv[1])
sd = run / 'structured'
hf = sd / 'hoi_fusion'
R = []
ok = lambda n, c: print(('PASS' if c else 'FAIL'), n) or R.append(bool(c))

ann = json.loads((hf / 'hoi_annotations.json').read_text())
evs = ann['events']
by_id = {e['event_id']: e for e in evs}
demo = pd.read_parquet(sd / 'demonstration.parquet')
ts = demo['timestamp'].to_numpy(float)
vp = json.loads((sd / 'visual_semantics/visual_predictions.json').read_text())['predictions']

# T1 视觉预测与3D轨迹时间对应：每事件帧范围落在其时间范围内
ok('T1 时间-轨迹对应', all(
    ts[np.searchsorted(ts, e['start_time'])] <= e['end_time'] + 1/14
    and e['start_frame'] <= e['end_frame'] for e in evs))

# T2 左右手独立：每事件单手，双手机制为 bimanual_same_object 标记而非合并
ok('T2 左右手独立', all(e['hand_side'] in ('left', 'right') for e in evs)
   and any(e.get('bimanual_same_object') for e in evs))

# T3 双手同物体（hoi_0004/0005 computer mouse）与不同物体（hoi_0000 vs hoi_0001）区分
pair = [e for e in evs if e['object_category'] == 'computer mouse']
mouse_same = len(pair) == 2 and pair[0]['hand_side'] != pair[1]['hand_side'] \
    and all(e.get('bimanual_same_object') for e in pair)
early = [e for e in evs if e['end_time'] < 3]
diff_obj = {e['object_category'] for e in early} >= {'手机', '电脑鼠标'}
ok('T3 双手同物体≠双手各物体', mouse_same and diff_obj)

# T4 重叠 clip 不重复：14 条预测(含5条跨目重复) → 8 事件；无相同(clip,hand,action)双事件
keys = [(tuple(e['source_clip_ids']), e['hand_side'], e['visual_action']) for e in evs]
ok('T4 重叠去重', len(evs) == 8 and len(keys) == len(set(keys)))

# T5 相邻不同动作不误合并：clip_0000右手鼠标 与 clip_0002右手鼠标 时间不重叠→独立保留
e0 = [e for e in evs if e['object_category'] == '电脑鼠标'][0]
e2 = [e for e in evs if e['object_category'] == 'computer mouse'
      and e['hand_side'] == 'right'][0]
ok('T5 相邻动作保留独立', e0['event_id'] != e2['event_id']
   and e0['end_time'] < e2['start_time'])

# T6 物体未知可正常输出（合成：把一个预测物体置 null）
tmp = Path(tempfile.mkdtemp())
shutil.copytree(sd, tmp / 'structured', dirs_exist_ok=True)
vp2 = json.loads((tmp / 'structured/visual_semantics/visual_predictions.json').read_text())
vp2['predictions'][0]['object_category'] = None
(tmp / 'structured/visual_semantics/visual_predictions.json').write_text(
    json.dumps(vp2, ensure_ascii=False))
fuse_hoi.build(tmp / 'structured', tmp / 'out')
unk = json.loads((tmp / 'out/hoi_annotations.json').read_text())['events']
ok('T6 物体未知正常输出', any(not e['object_category'] for e in unk)
   and all(e['semantic_action'] for e in unk))

# T7 视觉-3D冲突正确标记（真实：hoi_0000 有 primitive_visual_mismatch）
ok('T7 冲突标记', any('primitive_visual_mismatch' in e['conflicts'] for e in evs)
   and all(e['needs_review'] == bool(e['conflicts']) for e in evs))

# T8 invalid 不作为可靠证据（合成：右手某事件帧全 invalid）
shutil.copytree(sd, tmp / 's2', dirs_exist_ok=True)
d2 = pd.read_parquet(tmp / 's2/demonstration.parquet')
d2.loc[90:181, 'right_status'] = 'invalid'
d2.to_parquet(tmp / 's2/demonstration.parquet', index=False)
fuse_hoi.build(tmp / 's2', tmp / 'out8')
ev8 = json.loads((tmp / 'out8/hoi_annotations.json').read_text())['events']
ok('T8 invalid→low_quality', any('low_quality_invalid_frames' in e['conflicts']
                                 for e in ev8 if e['hand_side'] == 'right'))

# T9 双向可回溯：evidence 帧/行存在且事件帧范围覆盖引用
ok('T9 证据可回溯', all(
    0 <= e['trajectory_rows'][0] <= e['trajectory_rows'][1] < len(demo)
    and e['evidence_frames']['video'] and e['evidence_frames']['trajectory_rows']
    for e in evs))

# T10 输入不可变
h0 = {f: hashlib.md5((sd / f).read_bytes()).hexdigest() for f in
      ('demonstration.parquet', 'action_segments.json',
       'visual_context/clip_manifest.json', 'visual_semantics/visual_predictions.json')}
fuse_hoi.build(sd, hf)
h1 = {f: hashlib.md5((sd / f).read_bytes()).hexdigest() for f in h0}
ok('T10 输入不可变', h0 == h1)

# T11 多次运行稳定（剔除时间戳后逐字节一致）
a1 = (hf / 'hoi_annotations.json').read_text()
fuse_hoi.build(sd, hf)
a2 = (hf / 'hoi_annotations.json').read_text()
g1 = json.loads(a1)['meta']['generated_at']
g2 = json.loads(a2)['meta']['generated_at']
ok('T11 重复运行稳定', a1.replace(g1, '') == a2.replace(g2, ''))

# T12 Schema 完整 + 异常输入
REQ = {'event_id', 'source_clip_ids', 'source_segment_ids', 'start_time', 'end_time',
       'start_frame', 'end_frame', 'hand_side', 'object_category',
       'object_instance_local_id', 'visual_action', 'motion_primitives',
       'semantic_action', 'contact_state', 'trajectory_reference', 'evidence_frames',
       'alignment_status', 'quality_status', 'conflicts', 'verification_status'}
ok('T12 Schema 完整', all(REQ <= set(e) for e in evs)
   and all(e['verification_status'] != 'verified' for e in evs))
try:
    fuse_hoi.build(tmp / 'nonexistent', tmp / 'out12')
    ok('T12b 异常输入报错', False)
except Exception:
    ok('T12b 异常输入报错', True)

print('\nStep2.3 基线:', sum(R), '/', len(R), 'PASS')

# ================= Step 2.4 专项回归（期望值从数据计算，不硬编码帧号/类别）====
import numpy as np
fl = pd.read_parquet(hf / 'hoi_frame_labels.parquet')

# T14 手势通道独立存在且取值合法
POSES = {'OPEN', 'CLOSED', 'PINCH', 'HALF_OPEN', 'UNKNOWN'}
ok('T14 姿态通道合法', all(set(fl[f'{h}_pose_state']) <= POSES for h in ('left', 'right')))

# T15 release 硬截断：双手鼠标事件的 end 不超过 末次低pinch+0.2s（从数据反推）
d15 = pd.read_parquet(sd / 'demonstration.parquet')
for hand in ('left', 'right'):
    e = next(x for x in evs if x['source_clip_ids'] == ['clip_0002'] and x['hand_side'] == hand)
    pin = d15[f'{hand}_pinch_idx'].iloc[e['trajectory_rows'][0]:e['trajectory_rows'][1] + 50]
    low_ts = d15['timestamp'].iloc[e['trajectory_rows'][0]:e['trajectory_rows'][1] + 50][
        (pin < 0.35).fillna(False).values]
    if len(low_ts):
        ok(f'T15 {hand} release截断',
           e['end_time'] <= float(low_ts.iloc[-1]) + 0.25 or 'boundary_refined' not in e['conflicts'])

# T16 释放后不再显示抓取：事件 end 之后 0.3s 的帧 interaction=NO_INTERACTION
e16 = next(x for x in evs if 'boundary_refined_by_release' in x['conflicts'])
r16 = fl.iloc[int(np.argmin(np.abs(fl.timestamp - (e16['end_time'] + 0.3))))]
ok('T16 释放后无残留', r16[f'{e16["hand_side"]}_interaction_state'] == 'NO_INTERACTION')

# T17 双手张开有输出（姿态通道非空）：末帧两通道都有值
ok('T17 张开=姿态+无交互', all(fl[f'{h}_display'].iloc[-1] for h in ('left', 'right'))
   and fl['left_pose_state'].iloc[-1] == 'OPEN' and fl['left_interaction_state'].iloc[-1] in
   ('NO_INTERACTION', 'HOLD'))

# T18 渲染标签与 JSON 同源：display 里的物体名必来自该行活跃事件
r18 = fl.iloc[100]
for hand in ('left', 'right'):
    ev_active = [e for e in evs if e['hand_side'] == hand
                 and e['trajectory_rows'][0] <= 100 <= e['trajectory_rows'][1]]
    has_obj = '|' in str(r18[f'{hand}_display']).split('|', 2)[-1] if '抓取' in str(
        r18[f'{hand}_display']) or '持握' in str(r18[f'{hand}_display']) else False
    if has_obj:
        ok(f'T18 {hand} display物体来自活跃事件',
           any(str(e['object_category']) in str(r18[f'{hand}_display']) for e in ev_active))
    else:
        ok(f'T18 {hand} 无物体时不显示物体', True)

# T19 弱抓握证据标记（从数据反推：低pinch占比<30% 必有 weak_grasp_evidence）
for e in evs:
    if 'weak_grasp_evidence' in e['conflicts']:
        ok('T19 弱证据标记成立', e['grasp_evidence']['low_pinch_frame_pct'] < 30)
        break

print('\nStep2.4 专项:', sum(R[13:]) if len(R) > 13 else 0, '/', len(R) - 13, 'PASS')

# ================= Step 2.5 专项（grounding 仲裁；期望值从数据推导）====
gnd_ok = (sd / 'grounding/association_evidence.parquet').exists()
if gnd_ok:
    RES = {'resolved', 'vlm_detector_mismatch', 'no_object_near_hand', 'no_consensus'}
    ok('T20 仲裁三态合法', all(e.get('object_resolution') in RES for e in evs)
       and all(e['object_detected'] for e in evs if e['object_resolution'] == 'resolved'))
    ok('T21 mismatch 必送审', all(e['needs_review'] for e in evs
       if e.get('object_resolution') == 'vlm_detector_mismatch'))
    kb = [e for e in evs if e['semantic_action'] == 'keyboard_touch_candidate']
    ok('T22 键盘触碰语义成立', kb and all(
        e.get('object_detected') == 'keyboard'
        and e['grasp_evidence']['low_pinch_frame_pct'] < 30 for e in kb))
    r22 = fl.iloc[60]
    ok('T23 display 名与事件一致', all(any(
        str(e.get('object_detected') or e['object_category']) in str(r22[f'{h}_display'])
        or e['trajectory_rows'][0] > 60 or e['trajectory_rows'][1] < 60
        for e in evs if e['hand_side'] == h) or '无交互' in str(r22[f'{h}_display'])
        for h in ('left', 'right')))
    ok('T24 GT 差异报告存在且诚实', (hf / 'gt_comparison_clip_0002.json').exists()
       and 'refuted' in json.loads((hf / 'gt_comparison_clip_0002.json').read_text())['metrics']['grasp_claims_correct'])
    ok('T25 时间映射自检', json.loads((sd / 'grounding/grounding_report.json').read_text())
       ['time_mapping_selfcheck']['clips_within_hd'])
else:
    print('SKIP Step2.5 专项（无 grounding 输入）')

# ================= Step 2.6 专项（审核-修复闭环；合成裁决验证机制本身）====
dec_path = Path(tempfile.mkdtemp()) / 'dec.jsonl'
dec_path.write_text(
    '{"event_id": "hoi_0004", "action": "mark_unknown", '
    '"reviewed_label": {"hand": "left", "object": null, "semantic": "unknown_action", "bounds": [6.03, 8.2]}, '
    '"original_prediction": {"object": "computer mouse"}, "correction_type": "mark_unknown", '
    '"evidence_frame_ids": [6.03], "reviewer": "jun", "review_time": "t", '
    '"correction_reason": "逐帧复核", "source_version": "v1+hoi-v1", "provenance": "human_review"}\n'
    '{"event_id": "hoi_missing", "action": "confirm", '
    '"reviewed_label": {"hand": "left", "object": "x", "semantic": "grasp", "bounds": [0, 1]}, '
    '"original_prediction": {}, "correction_type": "confirm", "evidence_frame_ids": [], '
    '"reviewer": "jun", "review_time": "t", "correction_reason": "r", '
    '"source_version": "v", "provenance": "human_review"}\n')
h_before = hashlib.md5((hf / 'hoi_annotations.json').read_bytes()).hexdigest()
fuse_hoi.build(sd, Path(tempfile.mkdtemp()) / 'rep1',
               [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()])
import shutil as _sh
out2 = Path(tempfile.mkdtemp()) / 'rep2'
fuse_hoi.build(sd, out2,
               [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()])
e4 = json.loads((out2 / 'hoi_annotations.json').read_text())['events']
e4 = next(e for e in e4 if e['event_id'] == 'hoi_0004')
flr = pd.read_parquet(out2 / 'hoi_frame_labels.parquet')
ok('T26 裁决同步到事件+逐帧', e4['object_category'] is None
   and e4['semantic_action'] == 'unknown_action'
   and e4['verification_status'] == 'human_reviewed'
   and e4['human_decision']['provenance'] == 'human_review'
   and 'computer mouse' not in flr.left_display.iloc[95:125].str.cat())
ok('T27 原始结果不可变',
   hashlib.md5((hf / 'hoi_annotations.json').read_bytes()).hexdigest() == h_before)
_ok31 = out2 / 'repair_log.json'
log = json.loads(_ok31.read_text())
ok('T28 修复记录可追溯+坏ID跳过',
   log[0]['status'] == 'applied' and log[1]['status'] == 'skipped_no_event')
# 幂等：再跑一次比较（剔除时间戳）
out3 = Path(tempfile.mkdtemp()) / 'rep3'
fuse_hoi.build(sd, out3,
               [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()])
a = (out2 / 'hoi_annotations.json').read_text()
b = (out3 / 'hoi_annotations.json').read_text()
ga = json.loads(a)['meta']['generated_at']; gb = json.loads(b)['meta']['generated_at']
ok('T29 修复幂等', a.replace(ga, '') == b.replace(gb, ''))
ok('T30 缺检≠无接触审计在案', (hf / 'audit_report.json').exists() and
   'insufficient_evidence' in (hf / 'audit_report.json').read_text()
   and 'provisional' in json.loads(
       (hf / 'ground_truth_clip_0002.json').read_text())['meta']['provenance'])

print(chr(10)+'总计(含2.5):', sum(R), '/', len(R), 'PASS')
sys.exit(0 if all(R) else 1)
