#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.6 — HOI 人工审核界面（本地 Flask，全中文）。

每个待审事件一页：双目 HD 证据帧（叠手/物体框）、VLM 原文、grounding 证据、
冲突原因、时间范围；八种裁决动作 POST 后追加写入 hoi_fusion/
review_decisions.jsonl（provenance 完整，永不覆盖原始预测）。
修复由 fuse_hoi --decisions 消费，保证 JSON/逐帧标签/视频三产物同源。

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
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from review_workflow import apply_decisions, load_context

HAND_CN = {'left': '左手', 'right': '右手', 'unknown': '未知手'}
HAND_COLOR = {'left': (80, 220, 80), 'right': (60, 160, 255)}
SEM_CN = {'candidate_object_grasp': '抓取(候选)', 'candidate_object_transport': '搬运(候选)',
          'candidate_object_rotate': '旋转物体(候选)', 'candidate_approach': '接近(候选)',
          'candidate_release': '释放(候选)', 'candidate_lift': '提起(候选)',
          'gesture': '手势', 'keyboard_touch_candidate': '键盘触碰(候选)',
          'hand_rotation_no_object_evidence': '手部旋转(无物体证据)',
          'pick_up_phone': '拿起手机', 'hold_phone': '持握手机',
          'put_down_phone': '放下手机',
          'unknown_action': '未知动作', 'no_interaction': '无交互',
          'needs_semantic_review': '语义待复核'}
CONTACT_CN = {'visible_contact': '可见接触', 'probable_contact': '可能接触',
              'no_contact': '未见接触', 'unknown': '未知'}
CONFLICT_CN = {'time_boundary_mismatch': '时间边界不一致', 'primitive_visual_mismatch': '运动原语与视觉不符',
               'weak_grasp_evidence': '抓握证据弱', 'object_naming_mismatch': '双目物体叫法不同',
               'object_continuity_question': '物体连续性存疑',
               'object_vlm_detector_mismatch': '检测器与VLM物体不一致',
               'no_object_near_hand': '手附近未检出物体', 'camera_disagreement': '双目判断不一致',
               'boundary_refined_by_release': '边界已按释放动作收缩',
               'low_quality_invalid_frames': '无效帧过多', 'no_temporal_overlap': '视觉与3D无时间交集'}
RESOLUTION_CN = {'resolved': '已确认', 'vlm_detector_mismatch': 'VLM与检测冲突',
                 'no_object_near_hand': '附近无检出', 'no_consensus': '无共识'}
ACT_CN = [('confirm', '✅ 确认标签正确'), ('change_object', '✏️ 改物体'),
          ('change_action', '✏️ 改动作'), ('change_hand', '✏️ 改手别'),
          ('adjust_boundary', '⏱ 调整时间边界'), ('mark_no_interaction', '🚫 无交互(未接触物体)'),
          ('mark_unknown', '❓ 看不清/不确定'), ('keep_reviewing', '⏸ 保留待复核')]
OBJ_CN = {'mouse': '鼠标', 'keyboard': '键盘', 'cell phone': '手机', 'remote': '遥控器/手柄',
          'book': '书', 'cup': '杯子', 'scissors': '剪刀', 'laptop': '笔记本电脑'}
STYLE = """<style>
:root{--bg:#080d18;--panel:#121c2c;--panel2:#0c1523;--line:#273650;--text:#eaf1fa;--muted:#91a2b8;--blue:#4ca9ff;--green:#44d5a2;--orange:#f0a34a}
*{box-sizing:border-box}body{font-family:'Microsoft YaHei',Inter,system-ui,sans-serif;background:radial-gradient(circle at 10% 0,#14294a 0,transparent 28%),var(--bg);color:var(--text);margin:0 auto;padding:24px;width:min(1500px,100%);line-height:1.55}
h2{font-size:22px;margin:0 0 14px}p{color:var(--muted)}table{width:100%;border-collapse:separate;border-spacing:0;background:var(--panel2);border:1px solid var(--line);border-radius:12px;overflow:hidden}td,th{border-bottom:1px solid #21304a;padding:10px 12px;text-align:left;font-size:12px;vertical-align:top}th{background:#162238;color:#adc0d8;font-size:11px;letter-spacing:.04em}tr:last-child td{border-bottom:0}tr:hover td{background:#121f32}a{color:#72bfff}
.tag{display:inline-block;background:#322413;color:#ffc477;border:1px solid #65451c;border-radius:999px;padding:2px 7px;margin:2px;font-size:10px}.sec{background:linear-gradient(155deg,#142033,#101827);border:1px solid var(--line);border-radius:13px;padding:14px 16px;margin:11px 0;box-shadow:0 10px 28px #0002}
img{border:1px solid #344863;border-radius:9px;margin:4px;max-width:100%;height:auto}code,pre{background:#080f1b;border:1px solid #22314a;border-radius:7px;padding:5px;color:#c7d7e9}input,select{background:#0a1220;color:var(--text);border:1px solid #34445e;border-radius:8px;padding:8px 10px;margin:4px;outline:none}input:focus,select:focus{border-color:var(--blue);box-shadow:0 0 0 3px #4ca9ff22}button{background:linear-gradient(135deg,#1d86df,#5664ed);border:none;border-radius:9px;color:#fff;padding:10px 24px;font-size:14px;font-weight:700;cursor:pointer;margin-left:8px}.done{color:var(--green);font-weight:bold}.pending{color:var(--orange);font-weight:bold}.nav{display:inline-block;background:#24344c;color:#fff;border:1px solid #344963;border-radius:8px;padding:8px 13px;text-decoration:none}.nav:hover,button:hover{filter:brightness(1.12)}small{color:var(--muted)}
@media(max-width:760px){body{padding:12px}td,th{padding:7px;font-size:10px}table{display:block;overflow-x:auto}input{max-width:100%}}
</style>"""


def cn(v, table):
    return table.get(v, str(v)) if v is not None else '无'


def b64(img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def evidence_frames(sources, ev, hands, objs):
    out = []
    for k in (0, .5, 1):
        t = ev['start_time'] + (ev['end_time'] - ev['start_time']) * k
        pair = {}
        for cam_i, key in ((0, 'L'), (1, 'R')):
            source, offset = sources.get(key, (None, 0.0))
            if source is None:
                continue
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fi = int(round((offset + t) * fps))
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
                cv2.rectangle(fr, p1, p2, HAND_COLOR[h.hand], 2)
                cv2.putText(fr, HAND_CN[h.hand], p1, 0, 0.6, HAND_COLOR[h.hand], 2)
            for _, o in objs[(objs.camera == key)
                             & (np.abs(objs.frame_t - t) < 0.06)].iterrows():
                p1 = (int(o.x0 * W), int(o.y0 * H))
                p2 = (int(o.x1 * W), int(o.y1 * H))
                cv2.rectangle(fr, p1, p2, (0, 200, 255), 1)
                cv2.putText(fr, f"{OBJ_CN.get(o.cls, o.cls)}#{o.track_id}",
                            p1, 0, 0.5, (0, 200, 255), 1)
            pair[key] = b64(fr)
        out.append({'t': round(t, 2), 'imgs': pair})
    return out


def make_app(sd: Path, export: Path, context_path: Path | None = None):
    app = Flask(__name__)
    hf = sd / 'hoi_fusion'
    ann = json.loads((hf / 'hoi_annotations.json').read_text())
    events = ann['events']
    hands = pd.read_parquet(sd / 'grounding/hand_boxes.parquet')
    objs = pd.read_parquet(sd / 'grounding/object_tracks.parquet')
    alignment = json.loads((sd / 'visual_context/video_alignment.json').read_text())
    m = re.search(r'_rig_([\d.]+)s_', Path(alignment['videos']['left']['path']).name)
    offset = float(m.group(1)) if m else 0.0
    # 优先回源到 30fps 高清视频；源目录不可用时使用已严格对齐的窗口视频。
    if (export / 'camera0.mp4').exists():
        sources = {'L': (export / 'camera0.mp4', offset),
                   'R': (export / 'camera1.mp4', offset)}
    else:
        vids = alignment.get('videos', {})
        sources = {
            'L': (Path(vids['left']['path']), 0.0) if vids.get('left') else (None, 0.0),
            'R': (Path(vids['right']['path']), 0.0) if vids.get('right') else (None, 0.0),
        }
    decision_path = hf / 'review_decisions.jsonl'
    repair_lock = threading.Lock()

    def latest_decisions():
        latest = {}
        if decision_path.exists():
            for line in decision_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    dec = json.loads(line)
                    latest[dec['event_id']] = dec
        return latest

    @app.get('/')
    def index():
        decided = latest_decisions()
        rows = ''
        for i, e in enumerate(events):
            tags = ''.join("<span class='tag'>" + CONFLICT_CN.get(c, c) + '</span>'
                           for c in e['conflicts']) or '—'
            det = cn(e.get('object_detected'), OBJ_CN) if e.get('object_detected') else '—'
            state = ("<span class='done'>已人工裁决</span>" if e['event_id'] in decided
                     else ("<span class='pending'>待核验</span>" if e['needs_review']
                           else "例行可复核"))
            rows += (f"<tr><td><a href='/event/{i}'>{e['event_id']}</a></td>"
                     f"<td>{e['start_time']}–{e['end_time']}s</td>"
                     f"<td>{HAND_CN.get(e['hand_side'], e['hand_side'])}</td>"
                     f"<td>{cn(e['object_category'], OBJ_CN)}"
                     f"<br><small>检测:{det}({cn(e.get('object_resolution'), RESOLUTION_CN)})</small></td>"
                     f"<td>{cn(e['semantic_action'], SEM_CN)}</td>"
                     f"<td><small>{tags}</small></td><td>{state}</td></tr>")
        pending = sum(e['event_id'] not in decided for e in events if e['needs_review'])
        version = load_context(context_path).get('review_version', 0) if context_path else 0
        return (f"{STYLE}<h2>HOI 人工复核 · 候选不是事实</h2>"
                f"<p><a class='nav' href='http://localhost:7861/'>← 返回主界面</a></p>"
                f"<div class='sec'>待核验 <b>{pending}</b> 个；已裁决 <b>{len(decided)}</b> 个；"
                f"当前人工版本 <b>v{version:03d}</b>。机器冲突与弱证据保持未知/待核验，"
                f"只有人工提交后才生成权威修复版。</div>"
                f"<table><tr><th>事件</th><th>时间范围</th><th>手</th>"
                f"<th>物体(大模型说 / 检测器判)</th><th>动作语义</th><th>冲突原因</th><th>状态</th></tr>"
                f"{rows}</table>"
                f"<p>点击事件号进入审核 → 看双目画面 → 表单提交裁决。"
                f"看不清就选「❓ 看不清/不确定」。</p>")

    @app.get('/event/<int:i>')
    def event_page(i):
        e = events[i]
        frames = evidence_frames(sources, e, hands, objs)
        imgs = ''
        for fr in frames:
            imgs += f"<div style='display:inline-block;vertical-align:top'><b>t={fr['t']}s</b><br>"
            for key, name in (('L', '左目'), ('R', '右目')):
                if fr['imgs'].get(key):
                    imgs += f"<div><small>{name}</small><br><img src='{fr['imgs'][key]}' width=430></div>"
            imgs += '</div>'
        ge = e.get('grounding', {}).get('per_class', {})
        ge_html = ''.join(
            f"<tr><td>{OBJ_CN.get(c, c)}</td><td>{v['dist_median']}</td>"
            f"<td>{v['near_ratio']}</td><td>{v['n']}</td></tr>" for c, v in ge.items()) \
            or '<tr><td colspan=4>无检出</td></tr>'
        opts = ''.join(f"<option value='{v}'>{t}</option>" for v, t in ACT_CN)
        conflicts = ''.join(f"<span class='tag'>{CONFLICT_CN.get(c, c)}</span>"
                            for c in e['conflicts']) or '无'
        gev = e.get('grasp_evidence', {})
        decided = latest_decisions().get(e['event_id'])
        prior = (f"<div class='sec done'>该事件已有人工裁决：{decided['action']} · "
                 f"{decided['reviewer']} · {decided['review_time']}。再次提交会生成新版本，"
                 f"并以最后一次裁决为准。</div>") if decided else ''
        hand_opts = ''.join(
            f"<option value='{h}' {'selected' if h == e['hand_side'] else ''}>{name}</option>"
            for h, name in HAND_CN.items())
        return f"""{STYLE}<h2>{e['event_id']} · {HAND_CN.get(e['hand_side'], e['hand_side'])} · {e['start_time']}–{e['end_time']}s</h2>
{prior}
<div class='sec'><b>① 大模型(VLM)的判断：</b>物体「{cn(e['object_category'], OBJ_CN)}」，
动作「{e['visual_action']}」，接触「{cn(e['contact_state'], CONTACT_CN)}」，
时间估计 {e.get('visual_time_boundaries')}</div>
<div class='sec'><b>② 3D 运动证据：</b>原语 {e['motion_primitives']}；
事件内捏合帧占比 {gev.get('low_pinch_frame_pct')}%（低=抓握证据弱），捏合中位 {gev.get('pinch_median')}</div>
<div class='sec'><b>③ 物体检测关联：</b>{cn(e.get('object_resolution'), RESOLUTION_CN)}
<table><tr><th>物体</th><th>距离(手对角线)</th><th>接近占比</th><th>样本帧</th></tr>{ge_html}</table>
<small>距离&lt;0.95 且接近占比&gt;0.25 才算关联成功</small></div>
<div class='sec'><b>④ 冲突原因：</b>{conflicts}</div>
<div class='sec'><b>⑤ 双目证据画面（绿=左手 橙=右手 黄=物体）：</b><br>{imgs}</div>
<div class='sec'><b>⑥ 你的裁决：</b>
<form method='post' action='/event/{i}'>
 裁决: <select name='action'>{opts}</select>（看图后选一个）<br><br>
 改手别时选: <select name='hand'>{hand_opts}</select><br>
 改物体时填: <input name='object' placeholder='如 手机 或 留空'>（无交互/不确定不用填；填 null 表示无物体）<br>
 改动作时填: <input name='semantic' placeholder='如 抓取/持握/释放/键盘触碰'>（仅改动作时填）<br>
 改边界时填: 起点 <input name='t0' size=5 value='{e['start_time']}'> 秒 ～
 终点 <input name='t1' size=5 value='{e['end_time']}'> 秒<br><br>
 理由(必填): <input name='reason' size=60 placeholder='例如：手悬在鼠标上方未接触，实际在敲键盘'><br>
 审核者(必填): <input name='reviewer' size=12 placeholder='你的名字'>
 <button type='submit'>提交裁决</button></form></div>
<a href='/'>← 返回队列</a>"""

    @app.post('/event/<int:i>')
    def decide(i):
        e = events[i]
        f = request.form
        if not f.get('reason') or not f.get('reviewer'):
            return '「理由」和「审核者」必填', 400
        valid_actions = {v for v, _ in ACT_CN}
        if f.get('action') not in valid_actions:
            return '无效的裁决动作', 400
        if f['action'] == 'change_object' and not f.get('object'):
            return '选择“改物体”时必须填写物体', 400
        if f['action'] == 'change_action' and not f.get('semantic'):
            return '选择“改动作”时必须填写动作', 400
        try:
            t0, t1 = float(f['t0']), float(f['t1'])
        except (KeyError, TypeError, ValueError):
            return '时间边界必须是数字', 400
        if not 0 <= t0 < t1:
            return '时间边界必须满足 0 ≤ 起点 < 终点', 400
        meta = ann.get('meta', {})
        dec = {
            'event_id': e['event_id'],
            'original_prediction': {
                'object': e['object_category'], 'hand': e['hand_side'],
                'semantic': e['semantic_action'],
                'bounds': [e['start_time'], e['end_time']]},
            'action': f['action'],
            'reviewed_label': {
                'hand': (f.get('hand') or e['hand_side']),
                'object': (None if f.get('object') == 'null' else
                           (f.get('object') or e['object_category'])),
                'semantic': (f.get('semantic') or e['semantic_action']),
                'bounds': [t0, t1]},
            'correction_type': f['action'],
            'evidence_frame_ids': [round(e['start_time'] + k *
                                         (e['end_time'] - e['start_time']), 2) for k in (0, .5, 1)],
            'reviewer': f['reviewer'],
            'review_time': datetime.datetime.now().isoformat(timespec='seconds'),
            'correction_reason': f['reason'],
            'source_version': (str(meta.get('fusion_rules_version', 'unknown')) + '+' +
                               str(meta.get('schema_version', 'unknown'))),
            'provenance': 'human_review'}
        with repair_lock:
            with decision_path.open('a', encoding='utf-8') as fp:
                fp.write(json.dumps(dec, ensure_ascii=False) + '\n')
            if context_path is None:
                return ('已记录裁决，但当前以旧式参数启动，缺少重渲染上下文。'
                        '请从 7861 主界面进入人工复核。', 409)
            try:
                updated, _ = apply_decisions(context_path)
            except Exception as exc:
                return (f"{STYLE}<h3>⚠️ 裁决已安全记录，但自动回写失败</h3>"
                        f"<pre>{str(exc)}</pre>"
                        f"<p><a href='/event/{i}'>返回后重试提交</a></p>", 500)
        act_cn = dict(ACT_CN).get(f['action'], f['action'])
        return (f"{STYLE}<h3>✅ 已记录并完成三产物同步：{e['event_id']} → {act_cn}</h3>"
                f"<p>裁决人 {f['reviewer']}，理由「{f['reason']}」</p>"
                f"<p>人工版本 v{updated['review_version']:03d}：JSON、逐帧标签和主视频均已重建。"
                f" <a class='nav' href='http://localhost:7861/'>返回主界面查看新视频</a></p>"
                f"<p><a href='/'>← 继续审下一个</a></p>"
                f"<details><summary>查看原始记录(JSON)</summary><code>"
                f"{json.dumps(dec, ensure_ascii=False)}</code></details>")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--structured-dir', required=False)
    ap.add_argument('--export', required=False)
    ap.add_argument('--context', default=None,
                    help='review_context.json；从主界面启动时自动回写并重渲染')
    ap.add_argument('--port', type=int, default=7863)
    ap.add_argument('--parent-pid', type=int, default=None,
                    help='主界面进程 PID；主界面退出后复核服务自动退出')
    args = ap.parse_args()
    context_path = Path(args.context).resolve() if args.context else None
    if context_path:
        ctx = load_context(context_path)
        sd, export = Path(ctx['structured_dir']), Path(ctx['export'])
    elif args.structured_dir and args.export:
        sd, export = Path(args.structured_dir), Path(args.export)
    else:
        ap.error('请提供 --context，或同时提供 --structured-dir 与 --export')
    if args.parent_pid:
        def parent_watch():
            while True:
                time.sleep(2)
                try:
                    os.kill(args.parent_pid, 0)
                except OSError:
                    os._exit(0)
        threading.Thread(target=parent_watch, daemon=True).start()
    make_app(sd, export, context_path).run(
        host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
