#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.6 — HOI 人工审核界面（本地 Flask 轻量版）。

每个待审事件一页：双目 HD 证据帧（叠手/物体框）、VLM 原文、grounding 证据、
冲突原因、时间范围；审核者七种裁决（确认/改手/改物体/改动作/调边界/无交互/
未知/待复核）POST 后追加写入 hoi_fusion/review_decisions.jsonl（provenance
完整，永不覆盖原始预测）。修复由 fuse_hoi --decisions 消费，保证 JSON/逐帧
标签/视频三产物同源。

启动: python ego_pipeline/preprocessing/das/hoi_review_app.py \
          --structured-dir examples/gapfill_lyz/egocap_20261001_103828/structured \
          --export examples/egocap_das_export/egocap_20261001_103828_bag_0
浏览器 http://localhost:7863
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

COLORS = {'left': (80, 220, 80), 'right': (60, 160, 255)}
OBJ_COLOR = (0, 200, 255)
ACTIONS = ['confirm', 'change_hand', 'change_object', 'change_action',
           'adjust_boundary', 'mark_no_interaction', 'mark_unknown',
           'keep_reviewing']


def b64(img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def evidence_frames(export, ev, hands, objs, offset):
    """事件起点/中点/终点三时刻的双目 HD 帧 + 框叠加。"""
    out = []
    ts3 = [ev['start_time'] + (ev['end_time'] - ev['start_time']) * k for k in (0, .5, 1)]
    for t in ts3:
        pair = {}
        for cam_i, key in ((0, 'L'), (1, 'R')):
            fi = int(round((offset + t) * 30))
            cap = cv2.VideoCapture(str(export / f'camera{cam_i}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            cap.release()
            if not ok:
                continue
            H, W = fr.shape[:2]
            for _, h in hands[(hands.camera == key)
                              & (np.abs(hands.frame_t - t) < 0.06)].iterrows():
                p1 = (int(h.x0 * W), int(h.y0 * H))
                p2 = (int(h.x1 * W), int(h.y1 * H))
                cv2.rectangle(fr, p1, p2, COLORS[h.hand], 2)
                cv2.putText(fr, h.hand, p1[:2], 0, 0.5, COLORS[h.hand], 1)
            for _, o in objs[(objs.camera == key)
                             & (np.abs(objs.frame_t - t) < 0.06)].iterrows():
                p1 = (int(o.x0 * W), int(o.y0 * H))
                p2 = (int(o.x1 * W), int(o.y1 * H))
                cv2.rectangle(fr, p1, p2, OBJ_COLOR, 1)
                cv2.putText(fr, f"{o.cls}#{o.track_id}", p1[:2], 0, 0.45, OBJ_COLOR, 1)
            pair[key] = b64(fr)
        out.append({'t': round(t, 2), 'L': pair.get('L'), 'R': pair.get('R')})
    return out


def make_app(sd: Path, export: Path):
    app = Flask(__name__)
    hf = sd / 'hoi_fusion'
    ann = json.loads((hf / 'hoi_annotations.json').read_text())
    events = ann['events']
    hands = pd.read_parquet(sd / 'grounding/hand_boxes.parquet')
    objs = pd.read_parquet(sd / 'grounding/object_tracks.parquet')
    align = json.loads((sd / 'visual_context/video_alignment.json').read_text())
    import re
    m = re.search(r'_rig_([\d.]+)s_', Path(align['videos']['left']['path']).name)
    offset = float(m.group(1)) if m else 0.0

    @app.get('/')
    def index():
        rows = ''.join(
            f"<tr><td><a href='/event/{i}'>{e['event_id']}</a></td>"
            f"<td>{e['start_time']}–{e['end_time']}s</td><td>{e['hand_side']}</td>"
            f"<td>{e['object_category']} (det:{e.get('object_detected')})</td>"
            f"<td>{e['semantic_action']}</td><td>{','.join(e['conflicts'])}</td></tr>"
            for i, e in enumerate(events))
        return f"<h2>HOI 审核队列（{len(events)} 事件）</h2><table border=1>" \
               f"<tr><th>id</th><th>时间</th><th>手</th><th>物体(VLM/det)</th>" \
               f"<th>语义</th><th>冲突</th></tr>{rows}</table>" \
               f"<p>决议文件: {hf / 'review_decisions.jsonl'}"

    @app.get('/event/<int:i>')
    def event_page(i):
        e = events[i]
        frames = evidence_frames(export, e, hands, objs, offset)
        imgs = ''.join(
            f"<div style='display:inline-block'><div>t={f['t']}s</div>"
            + ''.join(f"<img src='{f[k]}' width=420>" for k in ('L', 'R') if f[k])
            + '</div>' for f in frames)
        g = json.dumps(e.get('grounding', {}).get('per_class', {}), ensure_ascii=False)
        opts = ''.join(f"<option>{a}</option>" for a in ACTIONS)
        return f"""<h2>{e['event_id']}</h2>
        <p>VLM: {e['visual_action']} / {e['object_category']} / contact={e['contact_state']}
        | 原语 {e['motion_primitives']} | 冲突 {e['conflicts']} | 抓握证据 {e.get('grasp_evidence')}</p>
        <p>grounding: <code>{g}</code></p>{imgs}
        <form method='post' action='/event/{i}'>
          裁决: <select name='action'>{opts}</select>
          手(改时): <select name='hand'><option>left</option><option>right</option></select>
          物体(改时): <input name='object' placeholder='mouse / null(无) / unknown'>
          动作(改时): <input name='semantic' placeholder='grasp/hold/release/touch/no_interaction/unknown'>
          边界(调时): <input name='t0' size=5 value='{e['start_time']}'> –
                      <input name='t1' size=5 value='{e['end_time']}'>
          理由(必填): <input name='reason' size=60>
          审核者: <input name='reviewer' size=12>
          <button>提交裁决</button></form><a href='/'>← 队列</a>"""

    @app.post('/event/<int:i>')
    def decide(i):
        e = events[i]
        f = request.form
        if not f.get('reason') or not f.get('reviewer'):
            return '理由与审核者必填', 400
        dec = {
            'event_id': e['event_id'],
            'original_prediction': {
                'object': e['object_category'], 'hand': e['hand_side'],
                'semantic': e['semantic_action'],
                'bounds': [e['start_time'], e['end_time']]},
            'action': f['action'],
            'reviewed_label': {
                'hand': f.get('hand') or e['hand_side'],
                'object': (None if f['object'] in ('null', '') else
                           (f['object'] or e['object_category'])),
                'semantic': f['semantic'] or e['semantic_action'],
                'bounds': [float(f['t0']), float(f['t1'])]},
            'correction_type': f['action'],
            'evidence_frame_ids': [fr['t'] for fr in
                                   evidence_frames(export, e, hands, objs, offset)],
            'reviewer': f['reviewer'],
            'review_time': datetime.datetime.now().isoformat(timespec='seconds'),
            'correction_reason': f['reason'],
            'source_version': ann['meta']['fusion_rules_version'] + '+' +
                              ann['meta']['schema_version'],
            'provenance': 'human_review'}
        with (hf / 'review_decisions.jsonl').open('a') as fp:
            fp.write(json.dumps(dec, ensure_ascii=False) + '\n')
        return jsonify(ok=True, decision=dec)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=True)
    ap.add_argument('--export', required=True)
    ap.add_argument('--port', type=int, default=7863)
    args = ap.parse_args()
    make_app(Path(args.structured_dir), Path(args.export)).run(
        host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
