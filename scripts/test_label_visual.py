#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""label_visual.py 自动测试（Step 2.2 第五部分）。用法：
  python scripts/test_label_visual.py <run_dir>
run_dir 例：examples/gapfill_lyz/egocap_20261001_103828
需先完成 label_visual.py 全量推理。"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ego_pipeline/preprocessing/das'))
import label_visual as lv

run = Path(sys.argv[1])
vc = run / 'structured/visual_context'
vs = run / 'structured/visual_semantics'
R = []
ok = lambda n, c: print(('PASS' if c else 'FAIL'), n) or R.append(bool(c))

# T1 schema/枚举/null 合法性
preds = json.loads((vs / 'visual_predictions.json').read_text())['predictions']
ENUM = {'hand_side': lv.ENUMS['hand_side'], 'visual_action': lv.ENUMS['visual_action'],
        'contact_state': lv.ENUMS['contact_state']}
ok('T1 schema/枚举合法', all(p[k] in v for p in preds for k, v in ENUM.items())
   and all(isinstance(p['object_category'], (str, type(None))) for p in preds))
ok('T1b null 归一化（无字符串"null"）',
   all(p['object_category'] != 'null' for p in preds))

# T2 相机绑定：L↔camera0 / R↔camera1
align = json.loads((vc / 'video_alignment.json').read_text())
ok('T2 相机绑定不混淆',
   'camera0' in align['videos']['left']['path'] and
   'camera1' in align['videos']['right']['path'])

# T3 证据帧落在 clip 视频范围内，且关键帧类别在场
man = {c['clip_id']: c for c in json.loads((vc / 'clip_manifest.json').read_text())}
def rng(p, s): return man[p['clip_id']]['video_frame_indices'][s]
ok('T3 证据帧在范围内', all(
   rng(p, 'left' if p['camera'] == 'L' else 'right')['first'] <= i <=
   rng(p, 'left' if p['camera'] == 'L' else 'right')['last']
   for p in preds for i in p['evidence_frame_indices']))
ok('T3b 关键帧参与证据', all(
   {'at_start', 'at_end'} <= set(p['evidence_types']) for p in preds))

# T4 证据帧来自实际输入（像素级）：at_start 帧 == Step2.1 keyframe 逐字节一致
import tempfile
p0 = next(p for p in preds if p['camera'] == 'L')
fi = next(i for i, k in zip(p0['evidence_frame_indices'], p0['evidence_types']) if k == 'at_start')
video = lv.resolve_video(align['videos']['left']['path'], vc)
tmp = Path(tempfile.mkdtemp())
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video),
                '-vf', f"select='eq(n\\,{fi})'", '-frames:v', '1',
                str(tmp / 'f.png')], check=True)
kf = vc / 'keyframes' / f"{p0['clip_id']}_at_start_L.png"
h1 = hashlib.md5(kf.read_bytes()).hexdigest()
h2 = hashlib.md5((tmp / 'f.png').read_bytes()).hexdigest()
ok('T4 证据帧像素一致(at_start)', h1 == h2)

# T5 重叠去重标记
dups = [p for p in preds if 'duplicate_of' in p]
overlap_pairs = [(a, b) for a in preds for b in preds
                 if a['prediction_id'] != b['prediction_id']
                 and min(a['clip_time_range'][1], b['clip_time_range'][1])
                 - max(a['clip_time_range'][0], b['clip_time_range'][0]) > 0
                 and (a['hand_side'], str(a['object_category']), a['visual_action'])
                 == (b['hand_side'], str(b['object_category']), b['visual_action'])]
ok('T5 重叠重复已标记' if overlap_pairs else 'T5 无重叠重复(空集合法)',
   len(dups) == len(overlap_pairs) // 2)

# T6 源文件不可变（与推理前 hash 对照——由调用方写入 .pre_hashes）
pre = run / 'structured/visual_semantics/.pre_hashes'
if pre.exists():
    before = json.loads(pre.read_text())
    after = {f: hashlib.md5((run / 'structured' / f).read_bytes()).hexdigest()
             for f in before}
    ok('T6 源文件不可变', before == after)

# T8 解析失败不编造（单元级）
hands, err = lv.parse_predictions('这不是JSON')
ok('T8 垃圾输入→None+原因', hands is None and err)
hands2, err2 = lv.parse_predictions('{"hands": [{"hand_side": "middle"}]}')
ok('T8b 非法枚举被拒', hands2 is None and 'hand_side' in err2)

# T9 vlm_raw 完整保存 + 可解析
raws = [json.loads(l) for l in (vs / 'vlm_raw.jsonl').read_text().splitlines() if l.strip()]
ok('T9 raw 完整(每clip×目各一条)', len(raws) == 2 * len(man)
   and all('raw_output' in r and 'model' in r and 'prompt_version' in r for r in raws))

print('\n总计:', sum(R), '/', len(R), 'PASS')
sys.exit(0 if all(R) else 1)
