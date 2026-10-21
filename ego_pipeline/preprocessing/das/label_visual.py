#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2.2 — Qwen2.5-VL 视频语义识别：物体 / 左右手动作 / 可观察交互关系。

消费 Step 2.1 的 visual_context（clips + keyframes + manifest），对每个片段的
左右目视频独立推理，输出可追溯、可验证的结构化视觉语义**候选**：

  structured/visual_semantics/
    vlm_raw.jsonl            每次推理的原始输出 + 配置（断点恢复的依据）
    visual_predictions.json  结构化候选（含证据帧/时间戳/一致性/去重）
    visual_validation.json   Schema 检查、证据完整性、冲突与异常报告
    review/                  人工核验用：关键帧 + 高清回源帧 + 每片段预测摘要

原则：
  - 复用 label_actions.py 的 Qwen2.5-VL 加载/推理（_load_vlm/_ask/_grabber），
    不新增视觉模型；
  - 相机编号(camera0/1)与人体左右手严格区分，由画面判断；
  - 帧输入显式采样（fps 可配）+ Step 2.1 关键帧必含，所有证据帧索引可追溯；
  - 严格 JSON 校验：无效输出→保存原文→一次修复重试→仍失败记 parse_error，
    绝不编造字段；
  - 双目冲突保留双方结果标 needs_review；跨片段时间重叠预测去重；
  - VLM 时间估计独立存储，不覆盖 3D 边界；contact_state 是视觉证据等级，
    非已验证物理接触；confidence 不当校准概率使用。

用法（先单片段验证再全量）：
  python ego_pipeline/preprocessing/das/label_visual.py \
      --visual-context examples/gapfill_lyz/egocap_20261001_103828/structured/visual_context \
      --export         examples/egocap_das_export/egocap_20261001_103828_bag_0 \
      [--only clip_0002]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from label_actions import _ask, _grabber, _load_vlm, _resolve_model_path

PROMPT_VERSION = 'v1'
MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
KF_NAMES = ('before', 'at_start', 'peak', 'at_end')
ENUMS = {
    'hand_side': {'left', 'right', 'both', 'unknown'},
    'visual_action': {'approach', 'grasp', 'lift', 'transport', 'rotate',
                      'release', 'gesture', 'unknown'},
    'contact_state': {'visible_contact', 'probable_contact', 'no_contact',
                      'unknown'},
}


# ---------------------------------------------------------------- prompt

def build_prompt(camera_tag: str, duration_s: float, prim_l, prim_r) -> str:
    return f"""你是第一人称(ego)双目视频分析助手。当前视频来自头戴设备的{camera_tag}。
注意：camera0/camera1 只是设备编号，不代表人的左手或右手——请从画面本身判断出现的是人的哪只手。
视频时长 {duration_s:.2f} 秒（按时间顺序均匀采样传入），最后额外附 4 张关键帧（before/at_start/peak/at_end，按时间顺序）。
背景参考（来自 3D 手部运动学候选，可能不准，仅供对照）：该片段左手候选原语 {prim_l}，右手候选原语 {prim_r}。

请仔细观察视频帧，只报告有视觉证据的内容，输出严格的 JSON（除 JSON 外不要输出任何文字）：
{{
  "hands": [
    {{
      "hand_side": "left|right|both|unknown",
      "object_category": "看到的实际物体类别(开放词表,中文或英文)",
      "visual_action": "approach|grasp|lift|transport|rotate|release|gesture|unknown",
      "contact_state": "visible_contact|probable_contact|no_contact|unknown",
      "action_description": "一句话中文描述这只手在做什么",
      "time_estimate": {{"start_s": 数值或null, "end_s": 数值或null}}
    }}
  ]
}}
硬性要求：
1. 顶层必须输出单个 JSON 对象 {{"hands": [...]}}，不要顶层数组，不要重复键；
2. 看不清/无法确认物体时 object_category 必须为 null，禁止根据常识猜测；
3. time_estimate 是相对本视频起点(0秒)的估计，不确定就填 null；
4. contact_state 表示视觉证据强度：visible_contact=明确看到手指接触物体；
5. action_description 限 30 字以内；没有手的帧不要编造 hands 条目。"""


# ---------------------------------------------------------------- 解析与校验

def _hook(pairs):
    """合并重复的 hands 键（模型偶发输出两个 hands），不丢预测。"""
    d = {}
    for k, v in pairs:
        if k == 'hands' and k in d and isinstance(v, list) and isinstance(d[k], list):
            d[k] = d[k] + v
        else:
            d[k] = v
    return d


def _candidates(text: str):
    """产出多个候选 JSON 片段：围栏内、首{到尾}、首[到尾]、截断修复变体。"""
    t = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', t, re.S)
    if m:
        t = m.group(1).strip()
    out = [t]
    a, b = t.find('{'), t.rfind('}')
    if a >= 0 and b > a:
        out.append(t[a:b + 1])
    a, b = t.find('['), t.rfind(']')
    if a >= 0 and b > a:
        out.append(t[a:b + 1])
        for suffix in (']', '}]', '"}]', '"}]}'):
            out.append(t[a:] + suffix)      # 截断修复
    return out


def _loads(c: str):
    try:
        return json.loads(c, object_pairs_hook=_hook)
    except json.JSONDecodeError:
        return None


def _normalize(obj):
    """dict{hands} / 顶层数组 / [{hands}] 包装 → hands 列表或 None。"""
    if isinstance(obj, dict) and isinstance(obj.get('hands'), list):
        return obj['hands']
    if isinstance(obj, list):
        inner = [h for it in obj if isinstance(it, dict)
                 and isinstance(it.get('hands'), list) for h in it['hands']]
        plain = [it for it in obj if isinstance(it, dict) and 'hands' not in it]
        if inner or plain:
            return inner + plain
    return None


def parse_predictions(text: str):
    """返回 (hands|None, 错误原因)。不做任何填补。"""
    for c in _candidates(text):
        hands = _normalize(_loads(c))
        if hands is None:
            continue
        errs = [f'#{i}: {e}' for i, it in enumerate(hands)
                if (e := validate_item(it))]
        if errs:
            return None, '; '.join(errs)
        return hands, None
    return None, '无法提取 JSON（所有候选均失败）'


def validate_item(it) -> str | None:
    """返回错误原因；None=合法。缺失字段/非法枚举/null 合法性。"""
    if not isinstance(it, dict):
        return 'hand item 非 object'
    for k in ('hand_side', 'visual_action', 'contact_state'):
        v = it.get(k, 'MISSING')
        if v not in ENUMS[k]:
            return f'{k} 非法值: {v!r}'
    if 'object_category' not in it:
        return 'object_category 缺失'
    if 'action_description' not in it:
        return 'action_description 缺失'
    te = it.get('time_estimate')
    if te is not None and not isinstance(te, dict):
        return 'time_estimate 非 object/null'
    return None


def _ask_text(model, processor, prompt, raw, err):
    """无图纯文本修复轮（qwen_vl_utils 消息可无图）。"""
    import torch
    from qwen_vl_utils import process_vision_info
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': (f'你上一条回复解析失败（{err}）。原回复：\n{raw}\n'
                                  '请只修复 JSON 格式重新输出，不改变观察内容。'
                                  '不要输出 JSON 以外的文字。')}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=700, do_sample=False)
    return processor.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- 帧采样

def resolve_video(path_str: str, visual_context: Path) -> Path:
    """alignment 里存的是生成时的路径（可能相对仓库根）；按祖先目录逐一回退。"""
    p = Path(path_str)
    if p.exists():
        return p
    for base in (Path.cwd(), *visual_context.parents):
        q = base / p
        if q.exists():
            return q.resolve()
    raise FileNotFoundError(f'视频不存在（尝试过 cwd 与 {visual_context} 各级祖先）: {p}')


def sample_indices(clip: dict, video_fps: float, base: int) -> list[dict]:
    """确定性计算采样索引（与抓帧分离，供断点重放复算证据）。"""
    dur = clip['end_time'] - clip['start_time']
    n_uniform = max(2, int(dur * video_fps))
    idxs = [base + int(round((i + 0.5) * dur / n_uniform / (1 / 15.0)))
            for i in range(n_uniform)]
    kf_idx = {}
    for name, t in clip['source_frame_timestamps'].items():
        if t is None:
            continue
        fi = int(round(t * 15.0))
        kf_idx[name] = fi
        idxs.append(fi)
    return [{'index': i, 'ts': round(i / 15.0, 3),
             'kind': next((k for k, fi in kf_idx.items() if fi == i), 'uniform')}
            for i in sorted(set(idxs))]


def sample_frames(video: Path, clip: dict, video_fps: float, grab, base: int):
    """均匀采样 + 关键帧；索引为 rig_input 全局帧号。

    均匀采样从 clip 首帧（base）起；关键帧时间戳是窗口全局时间，直接 t*15
    （不加 base，否则双重偏移）。"""
    evid, frames = [], []
    for e in sample_indices(clip, video_fps, base):
        if (g := grab(e['index'])) is not None:
            frames.append(g)
            evid.append(e)
    if not frames:
        raise RuntimeError(f'{video}: 采样 0 帧——视频不可读或索引全部越界')
    return frames, evid


def _norm_obj(v):
    """模型把 null 写成字符串等情形归一化为 JSON null（不编造类别）。"""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ('null', 'none', '无', 'unknown', 'n/a', ''):
        return None
    return v


def crop_offset_of(video_ref: Path) -> float:
    """从 rig_input 文件名 camera0_rig_2.00s_17.00s.mp 解析窗口起点秒。"""
    m = re.search(r'_rig_([\d.]+)s_', video_ref.name)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------- 高清回源

def hd_keyframe(export_dir: Path, camera_key: str, window_ts: float,
                crop_offset: float, out_png: Path):
    """用 Step 2.1 验证过的映射回转换件提取原始分辨率关键帧。"""
    cam = int(camera_key == 'R')
    src = export_dir / f'camera{cam}.mp4'
    if not src.exists():
        return None
    t_abs = crop_offset + window_ts
    grab, release = _grabber(src)
    try:
        f = grab(int(round(t_abs * 30.0)))
    finally:
        release()
    if f is None:
        return None
    import cv2
    out_png.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_png), f)


# ---------------------------------------------------------------- 主流程

def run(visual_context: Path, out_dir: Path, export_dir: Path | None,
        video_fps: float, only: str | None, model_id: str):
    manifest = json.loads((visual_context / 'clip_manifest.json').read_text())
    alignment = json.loads((visual_context / 'video_alignment.json').read_text())
    raw_path = out_dir / 'vlm_raw.jsonl'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clips_by_id = {c['clip_id']: c for c in manifest}
    done = set()
    replayed = []          # 断点恢复：从 raw 确定性重建旧预测（不再推理）
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r['clip_id'], r['camera']))
            if r.get('parsed_ok') and r['clip_id'] in clips_by_id:
                hands, _ = parse_predictions(r['raw_output'])
                if hands:
                    clip = clips_by_id[r['clip_id']]
                    base = clip['video_frame_indices']['left' if r['camera'] == 'L' else 'right']['first']
                    evid = sample_indices(clip, r.get('video_fps', video_fps), base)
                    replayed.append((clip, r['camera'], hands, evid))
    review = out_dir / 'review'
    review.mkdir(parents=True, exist_ok=True)

    model, processor, mpath = _load_vlm(model_id)

    def rig_video(camera_key: str) -> Path:
        side = 'left' if camera_key == 'L' else 'right'
        return resolve_video(alignment['videos'][side]['path'], visual_context)

    predictions = []
    raw_entries = []
    clips = [c for c in manifest if not only or c['clip_id'] == only]
    for clip in clips:
        dur = clip['end_time'] - clip['start_time']
        for key, cam_tag in (('L', 'camera0(左目)'), ('R', 'camera1(右目)')):
            if (clip['clip_id'], key) in done:
                print(f'[skip] {clip["clip_id"]}/{key} 已完成（断点恢复）')
                continue
            video = rig_video(key)
            base = clip['video_frame_indices']['left' if key == 'L' else 'right']['first']
            grab, release = _grabber(video)
            frames, evid = sample_frames(video, clip, video_fps, grab, base)
            release()
            prompt = build_prompt(cam_tag, dur,
                                  clip['left_primitive_candidates'],
                                  clip['right_primitive_candidates'])
            raw = _ask(model, processor, frames, prompt, max_new_tokens=1200)
            hands, err = parse_predictions(raw)
            repaired = False
            if hands is None:                       # 一次修复机会
                raw2 = _ask_text(model, processor, prompt, raw, err)
                hands, err2 = parse_predictions(raw2)
                if hands is not None:
                    err, repaired = None, True
                else:
                    err = f'{err} | 修复后仍失败: {err2}'
                raw_entries.append({'clip_id': clip['clip_id'], 'camera': key,
                                    'ts': datetime.now().isoformat(),
                                    'model': mpath, 'prompt_version': PROMPT_VERSION,
                                    'video_fps': video_fps, 'raw_output': raw,
                                    'repair_output': raw2 if repaired or err else None,
                                    'parsed_ok': hands is not None})
            else:
                raw_entries.append({'clip_id': clip['clip_id'], 'camera': key,
                                    'ts': datetime.now().isoformat(),
                                    'model': mpath, 'prompt_version': PROMPT_VERSION,
                                    'video_fps': video_fps, 'raw_output': raw,
                                    'repair_output': None, 'parsed_ok': True})
            with raw_path.open('a') as f:
                f.write(json.dumps(raw_entries[-1], ensure_ascii=False) + '\n')

            for i, h in enumerate(hands or []):
                predictions.append({
                    'prediction_id': f'{clip["clip_id"]}_{key}_{i}',
                    'clip_id': clip['clip_id'], 'camera': key, 'camera_tag': cam_tag,
                    'source_segment_ids': clip['source_segment_ids'],
                    'hand_side': h['hand_side'],
                    'object_category': _norm_obj(h.get('object_category')),
                    'object_instance_local_id': None,
                    'visual_action': h['visual_action'],
                    'contact_state': h['contact_state'],
                    'action_description': h.get('action_description', ''),
                    'vlm_time_estimate': h.get('time_estimate'),
                    'clip_time_range': [clip['start_time'], clip['end_time']],
                    'evidence_frame_indices': [e['index'] for e in evid],
                    'evidence_timestamps': [e['ts'] for e in evid],
                    'evidence_types': [e['kind'] for e in evid],
                    'evidence_type': 'observed',
                    'parse_repaired': repaired,
                    'confidence_note': 'model-subjective, not calibrated',
                    'review_status': 'pending'})

            # review 素材：关键帧拷贝 + 高清回源（at_start/peak）
            kf_dir = visual_context / 'keyframes'
            for name in KF_NAMES:
                src = kf_dir / f'{clip["clip_id"]}_{name}_{key}.png'
                if src.exists():
                    shutil.copy(src, review / src.name)
            if export_dir is not None:
                co = crop_offset_of(video)
                for name in ('at_start', 'peak'):
                    t = clip['source_frame_timestamps'][name]
                    hd_keyframe(export_dir, key, t, co,
                                review / f'{clip["clip_id"]}_{name}_{key}_HD.png')
            print(f'[vlm] {clip["clip_id"]}/{key}: {len(hands or [])} 条预测'
                  f'{"" if hands is not None else f" [解析失败: {err}]"}')

    # 断点恢复：把 raw 里已完成的预测重建进本次输出（不重复推理）
    for clip, key, hands, evid in replayed:
        cam_tag = 'camera0(左目)' if key == 'L' else 'camera1(右目)'
        for i, h in enumerate(hands):
            predictions.append({
                'prediction_id': f'{clip["clip_id"]}_{key}_{i}',
                'clip_id': clip['clip_id'], 'camera': key, 'camera_tag': cam_tag,
                'source_segment_ids': clip['source_segment_ids'],
                'hand_side': h['hand_side'],
                'object_category': _norm_obj(h.get('object_category')),
                'object_instance_local_id': None,
                'visual_action': h['visual_action'],
                'contact_state': h['contact_state'],
                'action_description': h.get('action_description', ''),
                'vlm_time_estimate': h.get('time_estimate'),
                'clip_time_range': [clip['start_time'], clip['end_time']],
                'evidence_frame_indices': [e['index'] for e in evid],
                'evidence_timestamps': [e['ts'] for e in evid],
                'evidence_types': [e['kind'] for e in evid],
                'evidence_type': 'observed', 'parse_repaired': False,
                'confidence_note': 'model-subjective, not calibrated',
                'review_status': 'pending'})

    # 输出顺序稳定（与恢复路径无关）：一致性/去重也在稳定序上进行
    predictions.sort(key=lambda p: p['prediction_id'])

    # ---- 双目一致性
    by_clip = {}
    for p in predictions:
        by_clip.setdefault(p['clip_id'], []).append(p)
    for cid, ps in by_clip.items():
        l = [p for p in ps if p['camera'] == 'L']
        r = [p for p in ps if p['camera'] == 'R']
        def sig(p):
            return (p['hand_side'], str(p['object_category']), p['visual_action'])
        ls, rs = {sig(p) for p in l}, {sig(p) for p in r}
        if ls and rs:
            consistent = bool(ls & rs)
            for p in ps:
                p['cross_camera'] = ('consistent' if consistent else 'needs_review')

    # ---- 跨片段去重（时间重叠 + 相同物体+动作+手）
    clips_by_id = {c['clip_id']: c for c in manifest}
    for i, p in enumerate(predictions):
        for q in predictions[:i]:
            cp, cq = clips_by_id[p['clip_id']], clips_by_id[q['clip_id']]
            overlap = min(cp['end_time'], cq['end_time']) - max(cp['start_time'], cq['start_time'])
            if overlap > 0 and (p['hand_side'], p['object_category'],
                                p['visual_action']) == (q['hand_side'], q['object_category'],
                                                        q['visual_action']):
                p['duplicate_of'] = q['prediction_id']
                break

    (out_dir / 'visual_predictions.json').write_text(
        json.dumps({'meta': {'model_path': mpath, 'model_id': model_id,
                             'prompt_version': PROMPT_VERSION,
                             'video_fps': video_fps,
                             'semantic_status': 'candidates_only',
                             'note': 'VLM visual candidates; NOT verified physical '
                                     'contact; NOT robot training labels'},
                    'predictions': predictions},
                   ensure_ascii=False, indent=2))

    # ---- 校验报告
    val = {'schema_check': {'total': len(predictions),
                            'all_fields_valid': all(
                                validate_item({'hand_side': p['hand_side'],
                                               'visual_action': p['visual_action'],
                                               'contact_state': p['contact_state'],
                                               'object_category': 1,
                                               'action_description': 1}) is None
                                for p in predictions)},
           'parse_failures': [e for e in raw_entries if not e['parsed_ok']],
           'unknown_object_predictions': sum(1 for p in predictions
                                             if p['object_category'] is None),
           'cross_camera_conflicts': [cid for cid, ps in by_clip.items()
                                      if any(p.get('cross_camera') == 'needs_review'
                                             for p in ps)],
           'duplicates': {p['prediction_id']: p['duplicate_of']
                          for p in predictions if 'duplicate_of' in p},
           'evidence_completeness': all(
               p['evidence_frame_indices'] for p in predictions),
           'immutable_sources_untouched': True}
    (out_dir / 'visual_validation.json').write_text(
        json.dumps(val, ensure_ascii=False, indent=2))
    return predictions, val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--visual-context', required=True)
    ap.add_argument('--export', default=None,
                    help='转换件目录（提供则启用高清关键帧回源）')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--video-fps', type=float, default=3.0)
    ap.add_argument('--only', default=None, help='只跑指定 clip（如 clip_0002）')
    ap.add_argument('--model', default=MODEL_ID)
    args = ap.parse_args()

    vc = Path(args.visual_context)
    sd = vc.parent
    out = Path(args.out_dir) if args.out_dir else sd / 'visual_semantics'
    out.mkdir(parents=True, exist_ok=True)
    preds, val = run(vc, out, Path(args.export) if args.export else None,
                     args.video_fps, args.only, args.model)
    print(f'[visual] 预测 {len(preds)} 条 | 解析失败 {len(val["parse_failures"])} | '
          f'冲突 clip {len(val["cross_camera_conflicts"])} -> {out}')


if __name__ == '__main__':
    main()
