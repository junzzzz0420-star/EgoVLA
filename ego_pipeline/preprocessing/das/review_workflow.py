#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HOI 人工复核闭环的运行上下文与自动回写。

机器候选始终保留在 ``structured/hoi_fusion``；人工裁决写入
``review_decisions.jsonl``，并重建 ``structured/hoi_repaired`` 与一个新的
``viz/<mode>+human_vNNN_*/multiview.mp4``。三种产物来自同一份裁决文件。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = Path(os.environ.get('EGO2LEROBOT_PY') or sys.executable)
CONTEXT_NAME = 'review_context.json'


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def save_context(path: Path, data: dict) -> dict:
    data = dict(data)
    data['updated_at'] = _dt.datetime.now().isoformat(timespec='seconds')
    _atomic_json(path, data)
    return data


def load_context(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def find_run_from_video(video: Path, allowed_root: Path) -> Path:
    video = video.resolve()
    allowed_root = allowed_root.resolve()
    if video.suffix.lower() != '.mp4' or not video.exists():
        raise ValueError('请选择一个存在的 MP4 运行结果')
    try:
        video.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError('运行结果不在允许的 examples 目录内') from exc
    for p in video.parents:
        if p.parent == allowed_root and (p / 'viz').is_dir():
            return p
        if (p / 'viz').is_dir() and (p / 'views').is_dir():
            return p
    raise ValueError('无法从视频定位对应运行目录')


def _infer_export(run: Path, mode: str) -> Path | None:
    root = REPO / 'examples' / 'egocap_das_export'
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and run.name == f'egocap_{d.name}_{mode}':
                return d.resolve()
    return None


def infer_context(run: Path, video: Path | None = None) -> tuple[Path, dict]:
    """为旧运行补建上下文；新运行由 run_multiview.py 直接写入。"""
    run = run.resolve()
    context_path = run / CONTEXT_NAME
    if context_path.exists():
        ctx = load_context(context_path)
        if video is not None and not ctx.get('main_video'):
            ctx['main_video'] = str(video.resolve())
            save_context(context_path, ctx)
        return context_path, ctx

    anns = list(run.glob('**/structured/hoi_fusion/hoi_annotations.json'))
    if not anns:
        raise ValueError('该结果没有 HOI 语义产物，请先勾选“主视频叠加中文红框语义”运行')
    ann = max(anns, key=lambda p: p.stat().st_mtime)
    sd = ann.parents[1]

    name = video.parent.name if video else ''
    mm = re.match(r'(mono|stereo|rig6)', name)
    if not mm:
        mm = re.search(r'_(mono|stereo|rig6)$', run.name)
    mode = mm.group(1) if mm else 'stereo'

    alignment = json.loads((sd / 'visual_context/video_alignment.json').read_text(
        encoding='utf-8'))
    left_path = Path(alignment['videos']['left']['path'])
    tm = re.search(r'_rig_([\d.]+)s_', left_path.name)
    seg_start = float(tm.group(1)) if tm else 0.0

    views = []
    for cam_dir in sorted((run / 'views').glob('camera*')):
        videos = sorted(cam_dir.rglob('file-000.mp4'))
        parquet = sorted(cam_dir.rglob('file-000.parquet'))
        if videos and parquet:
            views.append({'name': cam_dir.name, 'video': str(videos[0].resolve()),
                          'parquet': str(parquet[0].resolve())})
    if not views:
        raise ValueError('找不到重渲染所需的逐目视频/轨迹')

    fused = run / 'fused.parquet'
    if not fused.exists():
        fused = Path(views[0]['parquet'])
    export = _infer_export(run, mode)
    if export is None:
        raise ValueError('旧运行缺少源数据位置；请重新运行一次以生成复核上下文')

    ctx = {
        'schema_version': '1.0', 'run_dir': str(run), 'structured_dir': str(sd),
        'export': str(export), 'mode': mode, 'seg_start': seg_start,
        'fused': str(fused.resolve()), 'views': views,
        'actions': str(fused.with_suffix('.actions.json').resolve())
        if fused.with_suffix('.actions.json').exists() else None,
        'task_label': None, 'main_video': str(video.resolve()) if video else None,
        'candidate_annotations': str(ann.resolve()),
        'candidate_frame_labels': str((sd / 'hoi_fusion/hoi_frame_labels.parquet').resolve()),
        'hand_boxes': str((sd / 'grounding/hand_boxes.parquet').resolve()),
        'review_version': 0, 'status': 'pending_review',
    }
    return context_path, save_context(context_path, ctx)


def apply_decisions(context_path: Path) -> tuple[dict, str]:
    """应用全部人工裁决，原子更新上下文；失败时不宣称已完成。"""
    context_path = Path(context_path).resolve()
    ctx = load_context(context_path)
    sd = Path(ctx['structured_dir'])
    decisions = sd / 'hoi_fusion/review_decisions.jsonl'
    if not decisions.exists() or not decisions.read_text(encoding='utf-8').strip():
        raise ValueError('尚无人工裁决可应用')

    repaired = sd / 'hoi_repaired'
    commands: list[list[str]] = [[
        str(PY), str(HERE / 'fuse_hoi.py'), '--structured-dir', str(sd),
        '--decisions', str(decisions), '--out-dir', str(repaired),
    ]]
    version = int(ctx.get('review_version') or 0) + 1
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(ctx['run_dir']) / 'viz' / (
        f"{ctx['mode']}+human_v{version:03d}_{stamp}")
    view_specs = [f"{v['name']}={v['video']},{v['parquet']}" for v in ctx['views']]
    render = [
        str(PY), str(HERE / 'viz_multiview.py'), '--tag',
        f"{ctx['mode']}+human_v{version:03d}", '--views', *view_specs,
        '--fused', ctx['fused'], '--export', ctx['export'],
        '--seg-start', str(ctx['seg_start']), '--out-root',
        str(Path(ctx['run_dir']) / 'viz'), '--out-dir', str(out_dir),
        '--hoi-labels', str(repaired / 'hoi_frame_labels.parquet'),
        '--hand-boxes', ctx['hand_boxes'],
    ]
    if ctx.get('task_label'):
        render += ['--task-label', ctx['task_label']]
    commands.append(render)

    logs = []
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=str(REPO), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append('$ ' + ' '.join(cmd) + '\n' + proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f'自动回写失败（exit={proc.returncode}）\n{proc.stdout[-4000:]}')

    video = out_dir / 'multiview.mp4'
    required = [repaired / 'hoi_annotations.json',
                repaired / 'hoi_frame_labels.parquet', video]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('自动回写未生成完整产物：' + ', '.join(missing))

    ctx.update({
        'review_version': version, 'status': 'human_reviewed',
        'main_video': str(video.resolve()),
        'repaired_annotations': str(required[0].resolve()),
        'repaired_frame_labels': str(required[1].resolve()),
        'repair_log': str((repaired / 'repair_log.json').resolve()),
        'last_review_video': str(video.resolve()),
    })
    save_context(context_path, ctx)
    return ctx, '\n'.join(logs)
