# -*- coding: utf-8 -*-
"""语义（任务）标注工具：把人类/VLM 任务描述写进管线产物。

三种用法:
  1) 给一次 run 的全部 clip 打标注（界面用）:
     python label_tasks.py --run examples/das_app_runs/<录>_<模式> --task "拧螺丝组装工件"

  2) 给单个 lerobot_v3 目录打标注:
     python label_tasks.py --clip <path/to/lerobot_v3> --task "..."

  3) 生产模式 sidecar（写给源视频，之后跑管线可不带 --test，
     标注自动贯穿到数据集）:
     python label_tasks.py --video <path/to/xxx.mp4> --task "..." [--start 0 --end 74]

标注后重新聚合即生效:
     python ego_pipeline/tasks/lerobot_to_pi05.py --source <run目录> \
         --output <pi05目录> --action-mode absolute --from-source-task
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def update_clip_tasks(clip_root: Path, task: str) -> bool:
    """原位更新 lerobot_v3 的任务文本：tasks.parquet/jsonl + episodes 元数据。"""
    meta = Path(clip_root) / 'meta'
    ok = False
    pq_path = meta / 'tasks.parquet'
    if pq_path.exists():
        tbl = pq.read_table(pq_path)
        names = tbl.column_names
        n = tbl.num_rows
        data = {'task_index': tbl['task_index'].to_pylist() if 'task_index' in names
                else list(range(n))}
        data['task'] = [task] * n
        schema = pa.schema([pa.field('task_index', pa.int64()),
                            pa.field('task', pa.string())])
        pq.write_table(pa.table(data, schema=schema), pq_path)
        ok = True
    jl = meta / 'tasks.jsonl'
    if jl.exists():
        rows = [json.loads(l) for l in jl.read_text(encoding='utf-8').splitlines() if l.strip()]
        for r in rows:
            r['task'] = task
        jl.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
                      encoding='utf-8')
        ok = True
    # episodes 元数据（聚合端的第一优先级来源）
    for epq in sorted(meta.glob('episodes/chunk-*/*.parquet')):
        t = pq.read_table(epq)
        if 'tasks' in t.column_names:
            d = {c: t[c].to_pylist() for c in t.column_names}
            d['tasks'] = [[task] for _ in d['tasks']]
            pq.write_table(pa.table(d, schema=t.schema), epq)
            ok = True
    for ejl in sorted(meta.glob('episodes*.jsonl')):
        rows = [json.loads(l) for l in ejl.read_text(encoding='utf-8').splitlines() if l.strip()]
        for r in rows:
            if isinstance(r.get('tasks'), list) and r['tasks']:
                r['tasks'] = [task]
        ejl.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
                       encoding='utf-8')
    return ok


def write_video_sidecar(video: Path, task: str, start: int, end: int) -> Path:
    """写 {stem}_vlm_label.json（管线扫描自动识别，生产模式无需 --test）。"""
    seg = [{'task': task, 'start_frame': start, 'end_frame': end, 'main_type': None}]
    out = video.parent / f'{video.stem}_vlm_label.json'
    out.write_text(json.dumps(seg, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def label_run(run_dir: Path, task: str) -> int:
    n = 0
    for clip in sorted(Path(run_dir).rglob('lerobot_v3')):   # clip 即 lerobot_v3 目录
        if update_clip_tasks(clip, task):
            n += 1
    # 界面聚合产物（若已有 pi05）也刷新其 meta
    for info in sorted(Path(run_dir).rglob('pi05/meta/tasks.jsonl')):
        rows = [json.loads(l) for l in info.read_text(encoding='utf-8').splitlines() if l.strip()]
        for r in rows:
            r['task'] = task
        info.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
                        encoding='utf-8')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', help='一次 run 的输出目录（批量标注其全部 clip）')
    ap.add_argument('--clip', help='单个 lerobot_v3 目录')
    ap.add_argument('--video', help='源视频路径（写生产 sidecar）')
    ap.add_argument('--task', required=True, help='任务描述（语言指令）')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    args = ap.parse_args()

    if args.run:
        n = label_run(Path(args.run), args.task)
        print(f'[label] 更新 {n} 个 clip 的任务文本: {args.task!r}')
    elif args.clip:
        ok = update_clip_tasks(Path(args.clip), args.task)
        print(f'[label] {"OK" if ok else "未找到 tasks 文件"}: {args.clip}')
    elif args.video:
        v = Path(args.video)
        end = args.end if args.end is not None else 10 ** 9
        p = write_video_sidecar(v, args.task, args.start, end)
        print(f'[label] sidecar → {p}')
    else:
        ap.error('需要 --run / --clip / --video 之一')


if __name__ == '__main__':
    main()
