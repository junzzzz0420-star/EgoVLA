"""actors.join — join HaWoR and MoGe clip dependencies before SLAM."""

from __future__ import annotations

import ray
from ray.util.queue import Queue as RayQueue


@ray.remote
class ClipJoinStore:
    """State table for clip-level HaWoR/MoGe dependency joins."""

    def __init__(self):
        self._states: dict[str, dict] = {}

    def mark_hawor(self, item: dict, slam_q: RayQueue, stage_q: RayQueue | None = None) -> None:
        self._mark('hawor', item, slam_q, stage_q)

    def mark_moge(self, item: dict, slam_q: RayQueue, stage_q: RayQueue | None = None) -> None:
        self._mark('moge', item, slam_q, stage_q)

    def flush_errors(self, slam_q: RayQueue) -> int:
        emitted = 0
        for scene, state in list(self._states.items()):
            err = state.get('hawor_error') or state.get('moge_error')
            if not err:
                continue
            base = state.get('moge_item') or state.get('hawor_item') or state.get('item')
            if base is None:
                continue
            slam_q.put({**base, 'error': err})
            self._states.pop(scene, None)
            emitted += 1
        return emitted

    def pending_count(self) -> int:
        return len(self._states)

    def _mark(self, branch: str, item: dict, slam_q: RayQueue, stage_q: RayQueue | None) -> None:
        from ._utils import emit_stage_event

        scene = item['scene']
        state = self._states.setdefault(scene, {})
        state['item'] = item
        state[f'{branch}_done'] = True
        state[f'{branch}_item'] = item
        if item.get('error'):
            state[f'{branch}_error'] = item['error']
        elapsed = (
            float(item.get('t_moge') or 0.0) if branch == 'moge'
            else float(item.get('t_hawor12') or 0.0)
        )
        extra = {'elapsed': elapsed}
        if branch == 'hawor':
            extra.update({
                't_hawor_s1': float(item.get('t_hawor_s1') or 0.0),
                't_hawor_s2': float(item.get('t_hawor_s2') or 0.0),
            })
        emit_stage_event(
            stage_q, item, branch,
            'failed' if item.get('error') else 'done',
            error=item.get('error'),
            extra=extra,
        )

        if state.get('hawor_done') and state.get('moge_done'):
            hawor_item = state.get('hawor_item') or item
            moge_item = state.get('moge_item') or item
            events = []
            events.extend(hawor_item.get('events') or [])
            events.extend(moge_item.get('events') or [])
            err = state.get('hawor_error') or state.get('moge_error')
            t_moge = float(moge_item.get('t_moge') or 0.0)
            t_hawor12 = float(hawor_item.get('t_hawor12') or 0.0)
            t_moge_hawor = max(t_moge, t_hawor12)
            merged = {
                **moge_item,
                'hawor_video': hawor_item.get('hawor_video'),
                'detect_meta': hawor_item.get('detect_meta'),
                'fc_map': hawor_item.get('fc_map'),
                'cs_map': hawor_item.get('cs_map'),
                't_hawor_detect': hawor_item.get('t_hawor_detect', 0.0),
                't_hawor_s1': hawor_item.get('t_hawor_s1', 0.0),
                't_hawor_s2': hawor_item.get('t_hawor_s2', 0.0),
                't_hawor12': t_hawor12,
                't_moge': t_moge,
                't_moge_hawor': t_moge_hawor,
                'events': events,
                'error': err,
            }
            if not err:
                emit_stage_event(
                    stage_q, merged, 'moge_hawor', 'done',
                    extra={
                        'elapsed': t_moge_hawor,
                        't_moge': t_moge,
                        't_hawor': t_hawor12,
                    },
                )
            slam_q.put(merged)
            self._states.pop(scene, None)
