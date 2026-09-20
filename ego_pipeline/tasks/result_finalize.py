"""Internal helper."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import ray


def _clip_identity(item: dict) -> tuple[str, int]:
    """Return (parent_scene, clip_idx) used by manifest callbacks."""
    parent = item.get('_parent') if isinstance(item, dict) else None
    if parent:
        return parent.get('scene') or item.get('scene', ''), int(parent.get('clip_idx', 0))
    return item.get('scene', ''), int(item.get('clip_idx', 0))


@ray.remote
def _cpu_finalize_and_post(item: dict) -> dict:
    from steps.cpu.save_ops import save_raw_result, clean_and_save
    from steps.cpu.result_sidecars import write_lerobot3_sidecars

    scene    = item['scene']
    work_dir = Path(item['work_dir'])
    parent_scene, clip_idx = _clip_identity(item)

    timing_keys = (
        't_pre', 't_extract', 't_geo', 't_moge',
        't_hawor_detect', 't_hawor_s1', 't_hawor_s2', 't_hawor12', 't_moge_hawor',
        't_slam', 't_slam_align', 't_slam_droid_init', 't_slam_track', 't_slam_ba',
    )
    base = {
        'video':        item['video'],
        'scene':        scene,
        'parent_scene': parent_scene,
        'clip_idx':     clip_idx,
        'work_dir':     str(work_dir),
        'timings':      [],
        't_pre':        item.get('t_pre', 0.0),
        't_gpu':        item.get('t_geo', 0.0) + item.get('t_moge', 0.0)
                        + item.get('t_slam', 0.0),
    }
    for key in timing_keys:
        if key not in base:
            try:
                base[key] = float(item.get(key) or 0.0)
            except (TypeError, ValueError):
                base[key] = 0.0

    if item.get('error'):
        return {**base, 'pth': None, 'cleaned': None, 'error': item['error']}

    try:
        pred_result  = item['pred_result']
        result_dir   = work_dir / 'result'
        t = time.perf_counter()
        pth_path,     _ = save_raw_result(pred_result, result_dir, scene)
        cleaned_path, t_clean = clean_and_save(pred_result, result_dir, scene,
                                         cam_c2w=item.get('cam_c2w'),
                                         K=item.get('K'))
        sidecar = write_lerobot3_sidecars(
            pred_result=pred_result,
            result_dir=result_dir,
            scene_name=scene,
            source_video=item['video'],
            work_dir=work_dir,
            cam_c2w=item.get('cam_c2w'),
            K=item.get('K'),
            slam_hw=item.get('slam_hw'),
            fps=item.get('fps'),
            raw_pth_path=pth_path,
            cleaned_pth_path=cleaned_path,
            label_dir=item.get('label_dir'),
        )
        elapsed = time.perf_counter() - t
        from runtime.timing import log_timing
        log_timing('postprocess', scene, elapsed, t_clean=float(t_clean or 0.0))
        serial = (
            float(base.get('t_pre') or 0.0)
            + float(base.get('t_extract') or 0.0)
            + float(base.get('t_geo') or 0.0)
            + float(base.get('t_moge_hawor') or max(
                float(base.get('t_moge') or 0.0),
                float(base.get('t_hawor12') or 0.0),
            ))
            + float(base.get('t_slam') or 0.0)
            + elapsed
        )
        log_timing('clip_summary', scene, serial,
                   t_pre=base.get('t_pre', 0.0),
                   t_extract=base.get('t_extract', 0.0),
                   t_geo=base.get('t_geo', 0.0),
                   t_moge=base.get('t_moge', 0.0),
                   t_hawor=base.get('t_hawor12', 0.0),
                   t_moge_hawor=base.get('t_moge_hawor', 0.0),
                   t_slam=base.get('t_slam', 0.0),
                   t_clean=float(t_clean or 0.0))
        print(f'[pipeline]  {scene}; {elapsed:.1f}; {cleaned_path}.')
        return {**base, 'pth': pth_path, 'cleaned': cleaned_path,
                'lerobot3_manifest': sidecar.get('result_dir') and str(result_dir / 'lerobot3_manifest.json'),
                't_clean': float(t_clean or 0.0),
                't_post': elapsed, 'total_s': base['t_gpu'] + elapsed, 'error': None}
    except Exception:
        err = traceback.format_exc()
        print(f'[pipeline]  {scene}; {err.splitlines()[-1]}.')
        return {**base, 'pth': None, 'cleaned': None, 'error': err}


