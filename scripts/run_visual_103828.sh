#!/usr/bin/env bash
# Step 2.2 全链：单片段验证 → 全量推理 → 断点恢复验证 → 自动测试
set -euo pipefail
P=/home/jun/miniconda3/envs/ego2lerobot/bin/python
RUN=/home/jun/Ego2LeRobot/examples/gapfill_lyz/egocap_20261001_103828
VC=$RUN/structured/visual_context
VS=$RUN/structured/visual_semantics
EXP=/home/jun/Ego2LeRobot/examples/egocap_das_export/egocap_20261001_103828_bag_0
LOG=/tmp/visual_steps
cd /home/jun/Ego2LeRobot/ego_pipeline/preprocessing/das

rm -rf "$VS" && mkdir -p "$VS"
$P - "$RUN" <<'EOF'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
files = {}
for rel in ('demonstration.parquet', 'action_segments.json',
            'visual_context/clip_manifest.json', 'visual_context/video_alignment.json'):
    files[rel] = hashlib.md5((run / 'structured' / rel).read_bytes()).hexdigest()
(run / 'structured/visual_semantics/.pre_hashes').write_text(json.dumps(files))
EOF

echo "===== [1/4] 单片段 clip_0002 真实推理 ====="
$P label_visual.py --visual-context "$VC" --export "$EXP" --only clip_0002 >"$LOG.1" 2>&1
grep -E '^\[vlm\]|^\[visual\]' "$LOG.1"
$P - "$VS" <<'EOF'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1])/'visual_predictions.json').read_text())['predictions']
assert len(p) >= 1, '预测为空——推理或解析失败!'
assert all(len(x['evidence_frame_indices']) >= 8 for x in p), '证据帧为空!'
for x in p:
    print(' ', x['prediction_id'], '| hand', x['hand_side'], '| obj', repr(x['object_category']),
          '| act', x['visual_action'], '| contact', x['contact_state'],
          '| 证据帧', len(x['evidence_frame_indices']), '|', x['action_description'][:50])
EOF

echo "===== [2/4] 全量 4 片段（恢复跳过 clip_0002）====="
$P label_visual.py --visual-context "$VC" --export "$EXP" >"$LOG.2" 2>&1
grep -E '^\[vlm\]|^\[visual\]|^\[skip\]' "$LOG.2"

echo "===== [3/4] 断点恢复稳定性 ====="
cp "$VS/visual_predictions.json" /tmp/preds_before.json
$P label_visual.py --visual-context "$VC" --export "$EXP" --only clip_0000 >"$LOG.3" 2>&1
grep -cE '^\[skip\]' "$LOG.3" | xargs -I{} echo "  skip 行数: {}"
$P - <<'EOF'
import json
a = json.load(open('/tmp/preds_before.json'))
b = json.load(open('/home/jun/Ego2LeRobot/examples/gapfill_lyz/egocap_20261001_103828/structured/visual_semantics/visual_predictions.json'))
assert a == b, '恢复后输出变化'
print('  PASS 恢复后输出一致')
EOF

echo "===== [4/4] 自动测试 ====="
cd /home/jun/Ego2LeRobot
$P scripts/test_label_visual.py "$RUN"
echo ALL_VISUAL_DONE
