"""Stage timing logs and mean summaries for pipeline performance analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


STAGE_KEYS: list[tuple[str, str]] = [
    ('t_pre', 'preprocess.meta'),
    ('t_extract', 'preprocess.extract'),
    ('t_geo', 'GeoCalib'),
    ('t_moge', 'MoGe-2'),
    ('t_hawor_detect', 'HaWoR.detect'),
    ('t_hawor_s1', 'HaWoR.S1'),
    ('t_hawor_s2', 'HaWoR.S2'),
    ('t_hawor12', 'HaWoR'),
    ('t_moge_hawor', 'MoGe-2+HaWoR'),
    ('t_slam', 'MegaSaM/DROID-SLAM'),
    ('t_slam_align', 'MegaSaM.align'),
    ('t_slam_droid_init', 'DROID-SLAM.init'),
    ('t_slam_track', 'DROID-SLAM.track'),
    ('t_slam_ba', 'DROID-SLAM.BA'),
    ('t_clean', 'trajectory_clean'),
    ('t_post', 'postprocess'),
    ('t_lerobot', 'LeRobot'),
]

_KEY_TO_STAGE = {key: name for key, name in STAGE_KEYS}

DRIVER_STAGE_NAMES = {
    'geo': 'GeoCalib',
    'moge': 'MoGe-2',
    'hawor': 'HaWoR',
    'moge_hawor': 'MoGe-2+HaWoR',
    'megasam': 'MegaSaM/DROID-SLAM',
    'preprocess': 'preprocess.meta',
}


def log_timing(stage: str, scene: str, elapsed: float, **extra: Any) -> None:
    """Print a single-line stage timing record."""
    try:
        dt = float(elapsed or 0.0)
    except (TypeError, ValueError):
        dt = 0.0
    parts = [f'[timing] {stage}', f'scene={scene}', f'elapsed={dt:.2f}s']
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f'{key}={value:.2f}s' if key.startswith('t_') or 'elapsed' in key
                         else f'{key}={value:.2f}')
        else:
            parts.append(f'{key}={value}')
    print('  '.join(parts), flush=True)


def print_driver_stage_event(event: Mapping[str, Any]) -> None:
    """Reprint a worker stage event on the Ray driver so it shows in the main terminal."""
    if event.get('status') not in (None, 'done'):
        return
    elapsed = event.get('elapsed')
    if elapsed is None:
        return
    stage = DRIVER_STAGE_NAMES.get(str(event.get('stage') or ''), str(event.get('stage') or 'stage'))
    scene = event.get('item_scene') or event.get('scene') or '?'
    extra = {
        key: event[key]
        for key in (
            't_moge', 't_hawor', 't_hawor_s1', 't_hawor_s2',
            't_slam_align', 't_slam_droid_init', 't_slam_track', 't_slam_ba',
            'clip_idx',
        )
        if event.get(key) is not None
    }
    log_timing(stage, scene, elapsed, **extra)
    if stage == 'HaWoR':
        if event.get('t_hawor_s1'):
            log_timing('HaWoR.S1', scene, event['t_hawor_s1'])
        if event.get('t_hawor_s2'):
            log_timing('HaWoR.S2', scene, event['t_hawor_s2'])
    if stage == 'MegaSaM/DROID-SLAM':
        if event.get('t_slam_align'):
            log_timing('MegaSaM.align', scene, event['t_slam_align'])
        if event.get('t_slam_droid_init'):
            log_timing('DROID-SLAM.init', scene, event['t_slam_droid_init'])
        if event.get('t_slam_track'):
            log_timing('DROID-SLAM.track', scene, event['t_slam_track'])
        if event.get('t_slam_ba'):
            log_timing('DROID-SLAM.BA', scene, event['t_slam_ba'])


def print_clip_timing(result: Mapping[str, Any]) -> None:
    """Print per-clip stage timings after CPU postprocess returns to the driver."""
    scene = str(result.get('scene') or '?')
    if result.get('error'):
        return
    for key, name in STAGE_KEYS:
        if key == 't_lerobot':
            continue
        try:
            dt = float(result.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if dt > 0:
            log_timing(name, scene, dt)


def event_elapsed(events: Iterable[Mapping[str, Any]] | None, task: str) -> float:
    """Return elapsed seconds for the first matching event task name."""
    if not events:
        return 0.0
    for event in events:
        if event.get('task') != task:
            continue
        if event.get('elapsed') is not None:
            try:
                return max(0.0, float(event['elapsed']))
            except (TypeError, ValueError):
                pass
        start = event.get('t_start')
        end = event.get('t_end')
        if start is None or end is None:
            continue
        try:
            return max(0.0, float(end) - float(start))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _nonzero_mean(values: list[float]) -> tuple[float | None, int]:
    xs = [float(v) for v in values if v is not None and float(v) > 0.0]
    if not xs:
        return None, 0
    return sum(xs) / len(xs), len(xs)


def collect_clip_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'scene': result.get('scene'),
        'parent_scene': result.get('parent_scene') or result.get('scene'),
        'clip_idx': int(result.get('clip_idx', 0) or 0),
        'error': bool(result.get('error')),
    }
    for key, _ in STAGE_KEYS:
        try:
            row[key] = float(result.get(key) or 0.0)
        except (TypeError, ValueError):
            row[key] = 0.0
    return row


def summarize_timings(
    results: list[Mapping[str, Any]],
    wall_s: float,
    extra_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-clip timings into mean/sum counts, ignoring zeros."""
    rows = [collect_clip_row(r) for r in results]
    if extra_rows:
        rows.extend(collect_clip_row(r) for r in extra_rows)
    ok_rows = [r for r in rows if not r.get('error')]
    stages: dict[str, dict[str, Any]] = {}
    for key, name in STAGE_KEYS:
        mean, n = _nonzero_mean([float(r.get(key) or 0.0) for r in ok_rows])
        total = sum(float(r.get(key) or 0.0) for r in ok_rows)
        stages[name] = {
            'key': key,
            'mean_s': None if mean is None else round(mean, 3),
            'sum_s': round(total, 3),
            'n': n,
        }
    return {
        'n_clips': len(rows),
        'n_ok': len(ok_rows),
        'n_fail': len(rows) - len(ok_rows),
        'wall_s': round(float(wall_s), 3),
        'stages': stages,
        'clips': rows,
    }


def print_timing_summary(summary: Mapping[str, Any], sep: str = '=' * 60) -> None:
    print(f'\n{sep}')
    print(
        f'[timing] summary  ok={summary.get("n_ok", 0)}  '
        f'fail={summary.get("n_fail", 0)}  wall={summary.get("wall_s", 0):.1f}s'
    )
    stages = summary.get('stages') or {}
    print(f'{"stage":<24} {"mean":>10} {"n":>6} {"sum":>10}')
    for _, name in STAGE_KEYS:
        info = stages.get(name) or {}
        mean = info.get('mean_s')
        n = int(info.get('n') or 0)
        total = float(info.get('sum_s') or 0.0)
        mean_s = f'{mean:10.2f}' if mean is not None else f'{"—":>10}'
        print(f'{name:<24} {mean_s} {n:6d} {total:10.2f}')
    print(sep, flush=True)


def dump_timing_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[timing] wrote {out}', flush=True)
    return out


def merge_lerobot_timings(summary: dict[str, Any], lerobot_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Attach per-video LeRobot export timings onto an existing summary."""
    if not lerobot_rows:
        return summary
    mean, n = _nonzero_mean([float(r.get('t_lerobot') or 0.0) for r in lerobot_rows])
    total = sum(float(r.get('t_lerobot') or 0.0) for r in lerobot_rows)
    stages = dict(summary.get('stages') or {})
    stages['LeRobot'] = {
        'key': 't_lerobot',
        'mean_s': None if mean is None else round(mean, 3),
        'sum_s': round(total, 3),
        'n': n,
    }
    summary['stages'] = stages
    summary['lerobot'] = [
        {
            'scene': r.get('scene'),
            't_lerobot': float(r.get('t_lerobot') or 0.0),
            'returncode': r.get('returncode'),
        }
        for r in lerobot_rows
    ]
    return summary
