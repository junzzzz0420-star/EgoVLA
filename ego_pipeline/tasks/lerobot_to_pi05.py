"""Convert EgoPipeline LeRobot v3 exports into OpenPI Pi0.5 training data.

This follows the ego2robot ``pi05_ego_bimanual`` contract:

* ``observation.state`` is absolute 14-D EE in the first-camera gauge:
  ``lx,ly,lz,lrx,lry,lrz,left_gripper, rx,ry,rz,rrx,rry,rrz,right_gripper``
  Rotvec is a parallel-jaw frame (X palm-out, Y jaw opening, Z finger forward),
  remapped from MANO ``global_orient`` (not stored as raw MANO axes).
* ``action`` is next-frame relative EE (gripper stays absolute next value):
  ``dlx,dly,dlz,dlrx,dlry,dlrz,left_gripper, drx,dry,drz,drrx,drry,drrz,right_gripper``
* ``task`` / ``task_index`` come from ``--prompt`` (all episodes share it).
  If ``--prompt`` is omitted, every episode uses the placeholder
  ``<vlm_task_prompt>``.
* ``observation.images.ego`` is copied from the source LeRobot dataset

``--source`` may be one ``lerobot_v3`` root **or** a tree (e.g. Egocentric-100K).
Every nested ``lerobot_v3/meta/info.json`` is converted into one training episode
and merged into a single OpenPI LeRobot v2.1 dataset (14-D state/action).

Example::

    python ego_pipeline/tasks/lerobot_to_pi05.py \\
        --source /data/luogz/ego100k_lerobot/Egocentric-100K \\
        --output /data/dataset/pretrain_data/ego/lerobot_pi05 \\
        --overwrite \\
        --prompt "pick up the cup"

Relative vs absolute EE
-----------------------
LeRobot does **not** have to store relative EE. The working G2 dataset
``0908_video`` stores **absolute** next-frame joint / EE commands; the AgiBot
Pi0.5 config used on that data does **not** apply ``DeltaActions``. Official
OpenPI Aloha configs instead keep absolute actions in LeRobot and subtract
``action - state`` at train time. ego2robot / this exporter bake the relative
EE into the stored ``action`` column so ``pi05_ego_bimanual`` can consume it
without a delta transform.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as _R
from scipy.spatial.transform import Slerp

log = logging.getLogger("ego_pipeline.lerobot_to_pi05")

STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 1000
DEFAULT_VIDEO_KEY = "observation.images.ego"
PLACEHOLDER_TASK = "<vlm_task_prompt>"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "lerobot_pi05"

STATE_NAMES = [
    "lx", "ly", "lz", "lrx", "lry", "lrz", "left_gripper",
    "rx", "ry", "rz", "rrx", "rry", "rrz", "right_gripper",
]
ACTION_NAMES = [
    "dlx", "dly", "dlz", "dlrx", "dlry", "dlrz", "left_gripper",
    "drx", "dry", "drz", "drrx", "drry", "drrz", "right_gripper",
]

# OpenPose 21: thumb tip=4, index MCP=5, index tip=8, pinky MCP=17.
_PALM_A, _PALM_B = 5, 17
_PINCH_A, _PINCH_B = 4, 8

# MANO global_orient columns are not a parallel-jaw EE.
# Left:  X=fingers, Y=dorsum (out the back of the hand), Z ≈ thumb→pinky.
# Right: MANO is a mirrored left, so X=-fingers, Y=-dorsum.
# EE:    X=palm (out of the palm / toward the object), Y=jaw opening, Z=finger forward.
# R_ee = R_mano @ A, with det(A)=+1.
_MANO_TO_EE_LEFT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
_MANO_TO_EE_RIGHT = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


class EEActionConfig:
    def __init__(
        self,
        *,
        max_interp_gap: int = 3,
        mano_palm_width_m: float = 0.08,
        eps: float = 1e-8,
    ) -> None:
        if max_interp_gap < 0:
            raise ValueError("max_interp_gap must be non-negative")
        if mano_palm_width_m <= 0:
            raise ValueError("mano_palm_width_m must be positive")
        self.max_interp_gap = max_interp_gap
        self.mano_palm_width_m = mano_palm_width_m
        self.eps = eps


def _as_float(values, *, tail: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != object and tail != (-1,) and raw.ndim == 1 + len(tail):
        return raw.astype(dtype, copy=False)
    arr = np.stack([np.asarray(item, dtype=dtype).reshape(-1) for item in raw])
    if tail == (-1,):
        return arr
    return arr.reshape((arr.shape[0],) + tail)


def _as_bool(values, *, n: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == object:
        arr = np.asarray([
            bool(item) if np.ndim(item) == 0 else bool(np.asarray(item).reshape(-1)[0])
            for item in raw
        ], dtype=bool)
    else:
        arr = raw.astype(bool).reshape(-1)
    if n is not None and arr.shape != (n,):
        raise ValueError(f"expected bool length {n}, got {arr.shape}")
    return arr


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _rotvec(matrix: np.ndarray) -> np.ndarray:
    rot = _R.from_matrix(_orthonormalize(matrix).reshape(1, 3, 3))
    return rot.as_rotvec()[0].astype(np.float64)


def _mano_to_ee(rot_mano: np.ndarray, *, is_right: bool) -> np.ndarray:
    """Map a MANO world/cam orientation to the parallel-jaw EE frame."""
    axes = _MANO_TO_EE_RIGHT if is_right else _MANO_TO_EE_LEFT
    return _orthonormalize(np.asarray(rot_mano, dtype=np.float64).reshape(3, 3) @ axes)


def _interp_rotation(r0: np.ndarray, r1: np.ndarray, fraction: float) -> np.ndarray:
    key = _R.from_matrix(np.stack([_orthonormalize(r0), _orthonormalize(r1)]))
    return Slerp([0.0, 1.0], key)([float(fraction)])[0].as_matrix()


def _fill_hand(
    positions: np.ndarray,
    rotations: np.ndarray,
    grip: np.ndarray,
    observed: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, r, g = positions.copy(), rotations.copy(), grip.copy()
    valid = observed.copy()
    seen = np.flatnonzero(observed)
    if not len(seen):
        return p, r, g, valid

    for left, right in zip(seen[:-1], seen[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                fraction = k / (gap + 1)
                idx = left + k
                p[idx] = (1.0 - fraction) * p[left] + fraction * p[right]
                g[idx] = (1.0 - fraction) * g[left] + fraction * g[right]
                r[idx] = _interp_rotation(r[left], r[right], fraction)
                valid[idx] = True

    last = None
    for frame in range(len(observed)):
        if observed[frame]:
            last = frame
        elif not valid[frame] and last is not None:
            p[frame], r[frame], g[frame] = p[last], r[last], g[last]
    return p, r, g, valid


def _grip_from_kp21(joints: np.ndarray | None, observed: np.ndarray, cfg: EEActionConfig) -> np.ndarray:
    grip = np.zeros(len(observed), dtype=np.float64)
    if joints is None:
        return grip
    js = np.asarray(joints, dtype=np.float64).reshape(len(observed), 21, 3)
    for frame in np.flatnonzero(observed):
        if not np.all(np.isfinite(js[frame])):
            continue
        palm = float(np.linalg.norm(js[frame, _PALM_A] - js[frame, _PALM_B]))
        pinch = float(np.linalg.norm(js[frame, _PINCH_A] - js[frame, _PINCH_B]))
        grip[frame] = float(np.clip(pinch / max(palm, cfg.eps), 0.0, 1.0))
    return grip


def derive_bimanual_ee(
    *,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_rot: np.ndarray,
    right_rot: np.ndarray,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    left_kp21: np.ndarray | None = None,
    right_kp21: np.ndarray | None = None,
    config: EEActionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Derive 14-D absolute state and relative action in the first-camera gauge.

    ``cam_R`` / ``cam_t`` follow OpenCV ``x_cam = R @ x_world + t``. Only the
    first frame is used as a fixed world gauge, matching ego2robot.

    ``left_rot`` / ``right_rot`` are MANO ``global_orient`` matrices. They are
    remapped to a parallel-jaw EE: X palm-out, Y jaw opening, Z finger forward.
    """
    cfg = config or EEActionConfig()
    left_xyz = np.asarray(left_xyz, dtype=np.float64).reshape(-1, 3)
    right_xyz = np.asarray(right_xyz, dtype=np.float64).reshape(-1, 3)
    frames = left_xyz.shape[0]
    if right_xyz.shape[0] != frames:
        raise ValueError("left/right translation length mismatch")

    left_rot = np.asarray(left_rot, dtype=np.float64).reshape(frames, 3, 3)
    right_rot = np.asarray(right_rot, dtype=np.float64).reshape(frames, 3, 3)
    left_valid = np.asarray(left_valid, dtype=bool).reshape(frames)
    right_valid = np.asarray(right_valid, dtype=bool).reshape(frames)

    r = np.asarray(cam_R, dtype=np.float64)
    t = np.asarray(cam_t, dtype=np.float64)
    r0 = r[0] if r.ndim == 3 else r
    t0 = t[0] if t.ndim == 2 else t
    r0 = _orthonormalize(r0.reshape(3, 3))
    t0 = t0.reshape(3)

    state = np.zeros((frames, 2, 7), dtype=np.float64)
    matrices = np.tile(np.eye(3), (frames, 2, 1, 1))
    hand_valid = np.zeros((frames, 2), dtype=bool)

    for side, xyz, rot, observed, joints in (
        (0, left_xyz, left_rot, left_valid, left_kp21),
        (1, right_xyz, right_rot, right_valid, right_kp21),
    ):
        p = np.zeros((frames, 3), dtype=np.float64)
        rm = np.tile(np.eye(3), (frames, 1, 1))
        seen = observed.copy()
        for frame in range(frames):
            if not seen[frame]:
                continue
            if not (np.all(np.isfinite(xyz[frame])) and np.all(np.isfinite(rot[frame]))):
                seen[frame] = False
                continue
            p[frame] = r0 @ xyz[frame] + t0
            rm[frame] = _mano_to_ee(r0 @ rot[frame], is_right=(side == 1))
        grip = _grip_from_kp21(joints, seen, cfg)
        p, rm, grip, valid = _fill_hand(p, rm, grip, seen, cfg.max_interp_gap)
        state[:, side, :3] = p
        state[:, side, 3:6] = np.stack([_rotvec(value) for value in rm])
        state[:, side, 6] = grip
        matrices[:, side] = rm
        hand_valid[:, side] = valid

    action = np.zeros_like(state)
    for frame in range(frames - 1):
        for hand in range(2):
            if hand_valid[frame, hand] and hand_valid[frame + 1, hand]:
                action[frame, hand, :3] = state[frame + 1, hand, :3] - state[frame, hand, :3]
                action[frame, hand, 3:6] = _rotvec(
                    matrices[frame, hand].T @ matrices[frame + 1, hand]
                )
                action[frame, hand, 6] = state[frame + 1, hand, 6]
    action[-1, :, 6] = state[-1, :, 6]

    result = {
        "state": state.reshape(frames, STATE_DIM).astype(np.float32),
        "action": action.reshape(frames, ACTION_DIM).astype(np.float32),
        "hand_valid": hand_valid,
    }
    validate_ee_episode(result)
    return result


def derive_absolute_action(state: np.ndarray, hand_valid: np.ndarray) -> np.ndarray:
    """Store next-frame absolute EE as action (0908-style). Gripper is absolute."""
    state = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    valid = np.asarray(hand_valid, dtype=bool).reshape(-1, 2)
    action = np.zeros_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    for hand, sl in enumerate((slice(0, 7), slice(7, 14))):
        hold = ~(valid[:-1, hand] & valid[1:, hand])
        action[:-1][hold, sl] = state[:-1][hold, sl]
    return action


def validate_ee_episode(data: dict[str, np.ndarray]) -> None:
    state = np.asarray(data["state"])
    action = np.asarray(data["action"])
    valid = np.asarray(data["hand_valid"])
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError("state must have shape (T,14)")
    if action.shape != state.shape:
        raise ValueError("action must have shape (T,14)")
    if valid.shape != (len(state), 2):
        raise ValueError("hand_valid must have shape (T,2)")
    for name, value in data.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity")
    if np.any((state[:, [6, 13]] < 0) | (state[:, [6, 13]] > 1)):
        raise ValueError("state grip values must be in [0,1]")
    if np.any((action[:, [6, 13]] < 0) | (action[:, [6, 13]] > 1)):
        raise ValueError("action grip values must be in [0,1]")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def _load_task_map(root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    parquet = root / "meta" / "tasks.parquet"
    jsonl = root / "meta" / "tasks.jsonl"
    if parquet.exists():
        table = pq.read_table(parquet)
        names = set(table.column_names)
        text_col = "task" if "task" in names else "__index_level_0__" if "__index_level_0__" in names else None
        if "task_index" in names and text_col is not None:
            for index, text in zip(table["task_index"].to_pylist(), table[text_col].to_pylist()):
                tasks[int(index)] = "" if text is None else str(text)
            return tasks
        df = table.to_pandas()
        if "task_index" in df.columns:
            text = df["task"] if "task" in df.columns else df.index
            for index, value in zip(df["task_index"].tolist(), list(text)):
                tasks[int(index)] = "" if value is None else str(value)
            return tasks
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row.get("task") or "")
    return tasks


def _load_episode_meta(root: Path, info: dict[str, Any], video_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v3_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    if v3_dir.is_dir():
        paths = sorted(v3_dir.rglob("*.parquet"))
        if paths:
            table = pa.concat_tables([pq.read_table(path) for path in paths])
            rows = table.to_pylist()
    if not rows and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no episode metadata under {root}")
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def _resolve_video(root: Path, info: dict[str, Any], episode: dict[str, Any], video_key: str) -> Path:
    template = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    values = {
        "video_key": video_key,
        "chunk_index": int(episode.get(f"videos/{video_key}/chunk_index", episode.get("episode_chunk", 0))),
        "file_index": int(episode.get(f"videos/{video_key}/file_index", episode.get("episode_index", 0))),
        "episode_chunk": int(episode.get("episode_chunk", episode.get(f"videos/{video_key}/chunk_index", 0))),
        "episode_index": int(episode["episode_index"]),
    }
    try:
        path = root / template.format(**values)
    except KeyError:
        path = root / f"videos/chunk-{values['episode_chunk']:03d}/{video_key}/episode_{values['episode_index']:06d}.mp4"
    if path.exists():
        return path
    fallbacks = [
        root / "videos" / video_key / f"chunk-{values['chunk_index']:03d}" / f"file-{values['file_index']:03d}.mp4",
        root / "videos" / f"chunk-{values['episode_chunk']:03d}" / video_key / f"episode_{values['episode_index']:06d}.mp4",
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return path


def _resolve_data_table(root: Path, info: dict[str, Any], episode: dict[str, Any], cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    template = info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    values = {
        "chunk_index": int(episode.get("data/chunk_index", episode.get("episode_chunk", 0))),
        "file_index": int(episode.get("data/file_index", episode.get("episode_index", 0))),
        "episode_chunk": int(episode.get("episode_chunk", episode.get("data/chunk_index", 0))),
        "episode_index": int(episode["episode_index"]),
    }
    try:
        path = root / template.format(**values)
    except KeyError:
        path = root / f"data/chunk-{values['episode_chunk']:03d}/episode_{values['episode_index']:06d}.parquet"
    if not path.exists():
        alt = root / f"data/chunk-{values['episode_chunk']:03d}/episode_{values['episode_index']:06d}.parquet"
        if alt.exists():
            path = alt
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(path)
        cache[path] = pq.read_table(path).to_pandas()
    df = cache[path]
    if "episode_index" in df.columns:
        sliced = df[df["episode_index"] == int(episode["episode_index"])].copy()
    else:
        sliced = df.copy()
    if sliced.empty:
        raise ValueError(f"{path}: no rows for episode {episode['episode_index']}")
    if "frame_index" in sliced.columns:
        sliced = sliced.sort_values("frame_index")
    return sliced.reset_index(drop=True)


def _w2c_from_rows(df: pd.DataFrame, frames: int) -> tuple[np.ndarray, np.ndarray]:
    if "extrinsics_w2c" not in df.columns:
        return np.tile(np.eye(3), (frames, 1, 1)), np.zeros((frames, 3), dtype=np.float64)
    mats = _as_float(df["extrinsics_w2c"].to_numpy(), tail=(4, 4))
    return mats[:, :3, :3], mats[:, :3, 3]


def _world_from_camera_state(
    state: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = 0 if side == 0 else 61
    xyz_cam = state[:, base:base + 3]
    rot_cam = _R.from_euler("xyz", state[:, base + 3:base + 6]).as_matrix()
    xyz_world = np.einsum("tij,tj->ti", np.transpose(R_w2c, (0, 2, 1)), xyz_cam - t_w2c)
    rot_world = np.einsum("tji,tjk->tik", R_w2c, rot_cam)
    return xyz_world, rot_world


def _pose_finite(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=1) & np.isfinite(rot).reshape(len(xyz), -1).all(axis=1)


def _resolve_hand_valid(
    df: pd.DataFrame,
    frames: int,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_rot: np.ndarray,
    right_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_valid = np.zeros(frames, dtype=bool)
    right_valid = np.zeros(frames, dtype=bool)
    if "left_kept" in df.columns:
        left_valid = _as_bool(df["left_kept"].to_numpy(), n=frames)
        right_valid = _as_bool(df["right_kept"].to_numpy(), n=frames)
    elif "state_mask" in df.columns:
        mask = _as_float(df["state_mask"].to_numpy(), tail=(2,), dtype=np.float64)
        left_valid = mask[:, 0].astype(bool)
        right_valid = mask[:, 1].astype(bool)
    if left_valid.any() or right_valid.any():
        return left_valid, right_valid
    left_valid = _pose_finite(left_xyz, left_rot)
    right_valid = _pose_finite(right_xyz, right_rot)
    if left_valid.any() or right_valid.any():
        log.debug(
            "left_kept/right_kept/state_mask are all false; "
            "falling back to finite world-frame poses (%d left, %d right)",
            int(left_valid.sum()), int(right_valid.sum()),
        )
    return left_valid, right_valid


def _episode_ee_arrays(df: pd.DataFrame) -> dict[str, Any]:
    frames = len(df)
    R_w2c, t_w2c = _w2c_from_rows(df, frames)
    if "left_transl_world" in df.columns and "left_orient_world" in df.columns:
        left_xyz = _as_float(df["left_transl_world"].to_numpy(), tail=(3,))
        right_xyz = _as_float(df["right_transl_world"].to_numpy(), tail=(3,))
        left_rot = _as_float(df["left_orient_world"].to_numpy(), tail=(3, 3))
        right_rot = _as_float(df["right_orient_world"].to_numpy(), tail=(3, 3))
    elif "observation.state" in df.columns:
        state = _as_float(df["observation.state"].to_numpy(), tail=(-1,))
        if state.shape[1] < 122:
            raise ValueError(f"observation.state dim {state.shape[1]} < 122")
        left_xyz, left_rot = _world_from_camera_state(state, R_w2c, t_w2c, 0)
        right_xyz, right_rot = _world_from_camera_state(state, R_w2c, t_w2c, 1)
    else:
        raise ValueError("episode is missing both world-frame hands and observation.state")

    left_valid, right_valid = _resolve_hand_valid(
        df, frames, left_xyz, right_xyz, left_rot, right_rot,
    )

    left_kp21 = right_kp21 = None
    if "left_hand_kp21_world" in df.columns:
        left_kp21 = _as_float(df["left_hand_kp21_world"].to_numpy(), tail=(21, 3))
    if "right_hand_kp21_world" in df.columns:
        right_kp21 = _as_float(df["right_hand_kp21_world"].to_numpy(), tail=(21, 3))

    timestamps = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else None
    task_index = df["task_index"].to_numpy(dtype=np.int64) if "task_index" in df.columns else None
    return {
        "left_xyz": left_xyz,
        "right_xyz": right_xyz,
        "left_rot": left_rot,
        "right_rot": right_rot,
        "cam_R": R_w2c,
        "cam_t": t_w2c,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_kp21": left_kp21,
        "right_kp21": right_kp21,
        "timestamps": timestamps,
        "task_index": task_index,
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def _probe_video(path: Path) -> tuple[int, int, int, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return height, width, frames, "avc1"


def _vector_stats(array: np.ndarray) -> dict[str, list]:
    axis = 0 if array.ndim > 1 else None
    std = array.std(axis=axis)
    std = np.maximum(std, 1e-6)
    return {
        "min": np.asarray(array.min(axis=axis), dtype=np.float64).reshape(-1).tolist(),
        "max": np.asarray(array.max(axis=axis), dtype=np.float64).reshape(-1).tolist(),
        "mean": np.asarray(array.mean(axis=axis), dtype=np.float64).reshape(-1).tolist(),
        "std": np.asarray(std, dtype=np.float64).reshape(-1).tolist(),
        "count": [int(len(array))],
    }


def _scalar_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    std = float(values.std()) if len(values) else 1e-6
    return {
        "min": [float(values.min()) if len(values) else 0.0],
        "max": [float(values.max()) if len(values) else 0.0],
        "mean": [float(values.mean()) if len(values) else 0.0],
        "std": [max(std, 1e-6)],
        "count": [int(len(values))],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    text = str(prompt).strip()
    return text or None


def _episode_task_text(
    *,
    prompt: str | None,
    placeholder: str = PLACEHOLDER_TASK,
    episode: dict[str, Any] | None = None,
    task_map: dict[int, str] | None = None,
    arrays: dict[str, Any] | None = None,
    from_source_task: bool = False,
) -> str:
    """Use ``prompt`` for every episode when set; otherwise the placeholder.

    ``from_source_task`` only applies when ``prompt`` is omitted.
    """
    if prompt is not None:
        return prompt
    if from_source_task:
        episode = episode or {}
        task_map = task_map or {}
        arrays = arrays or {}
        tasks = episode.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks:
            text = str(tasks[0]).strip()
            if text:
                return text
        indices = arrays.get("task_index")
        if indices is not None and len(indices):
            text = task_map.get(int(indices[0]), "")
            if text and text.lower() != "null":
                return text
    return placeholder


def _is_lerobot_root(path: Path) -> bool:
    return (Path(path) / "meta" / "info.json").is_file()


def _pipeline_v3_hint(source: Path) -> str:
    manifest = source / "run_batch_manifest.json"
    if not manifest.is_file():
        return (
            "Pipeline conversion only works after clips finish successfully. "
            "Run `python -m cli pipeline --resume <this-dir> --retry_failed` first."
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Pipeline conversion only works after clips finish successfully."
    videos = payload.get("videos") or {}
    failed = [item for item in videos.values() if str(item.get("status", "")).lower() == "failed"]
    done = [item for item in videos.values() if str(item.get("status", "")).lower() in {"done", "ok", "success"}]
    first_error = ""
    for item in failed:
        text = str(item.get("error") or "").strip()
        if text:
            first_error = text.splitlines()[-1].strip()
            break
    extra = f" First error: {first_error}." if first_error else ""
    return (
        f"Pipeline wrote 0 LeRobot v3 datasets "
        f"({len(failed)}/{len(videos)} clips failed, {len(done)} done).{extra} "
        "Re-run with `--resume` and `--retry_failed` after fixing the pipeline error."
    )


def discover_lerobot_roots(source: Path) -> list[Path]:
    """Return one or many EgoPipeline ``lerobot_v3`` roots under ``source``.

    ``source`` may be a single LeRobot dataset (``meta/info.json`` at the root)
    or a pipeline tree containing nested ``lerobot_v3/`` directories.
    """
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if _is_lerobot_root(source):
        return [source]
    roots = sorted({info.parent.parent.resolve() for info in source.rglob("lerobot_v3/meta/info.json")})
    if not roots:
        roots = sorted({
            info.parent.parent.resolve()
            for info in source.rglob("meta/info.json")
            if _is_lerobot_root(info.parent.parent)
        })
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot v3 under {source} "
            f"(need meta/info.json, or */lerobot_v3/meta/info.json). "
            f"{_pipeline_v3_hint(source)}"
        )
    return roots


def _rel_source(tree: Path, clip: Path) -> str:
    try:
        return str(clip.resolve().relative_to(tree.resolve()))
    except ValueError:
        return str(clip.resolve())


def _chunk_index(episode_index: int, chunk_size: int = CHUNK_SIZE) -> int:
    return int(episode_index) // int(chunk_size)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_dataset_meta(
    output_root: Path,
    *,
    out_episodes: list[dict[str, Any]],
    out_stats: list[dict[str, Any]],
    task_to_index: dict[str, int],
    video_key: str,
    video_hw: tuple[int, int],
    fps: float,
    action_mode: str,
    repo_id: str,
    total_frames: int,
    task_prompt: str,
) -> None:
    height, width = video_hw
    n_ep = len(out_episodes)
    n_chunks = 1 if n_ep == 0 else _chunk_index(n_ep - 1) + 1
    out_info = {
        "codebase_version": "v2.1",
        "robot_type": "human_ego_pseudo_bimanual",
        "total_episodes": n_ep,
        "total_frames": int(total_frames),
        "total_tasks": len(task_to_index),
        "total_videos": n_ep,
        "total_chunks": n_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            video_key: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": float(fps),
                    "video.codec": "avc1",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ACTION_NAMES,
            },
            "observation.hand_valid": {
                "dtype": "bool",
                "shape": [2],
                "names": ["left", "right"],
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
        "repo_id": repo_id,
        "action_mode": action_mode,
        "task_prompt": task_prompt,
        "task_prompt_placeholder": PLACEHOLDER_TASK,
        "ee_layout": {
            "state": "absolute first-camera-frame EE [xyz, rotvec, gripper] left then right; rotvec is palm-out / jaw-open / finger-forward, not raw MANO",
            "action_relative": "next-minus-current EE; gripper is next absolute value",
            "action_absolute": "next-frame absolute EE, matching 0908_video storage style",
        },
    }
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "info.json").write_text(json.dumps(out_info, indent=2), encoding="utf-8")
    _write_jsonl(output_root / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": text} for text, index in task_to_index.items()],
    )
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", out_stats)


def convert(
    source_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    video_key: str = DEFAULT_VIDEO_KEY,
    action_mode: str = "relative",
    prompt: str | None = None,
    from_source_task: bool = False,
    max_interp_gap: int = 3,
    repo_id: str = "local/ego-pi05-bimanual",
    limit: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    if action_mode not in {"relative", "absolute"}:
        raise ValueError("action_mode must be 'relative' or 'absolute'")
    prompt = _normalize_prompt(prompt)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    roots = discover_lerobot_roots(source_root)
    log.info("found %d lerobot_v3 root(s) under %s", len(roots), source_root)
    if limit is not None:
        roots = roots[: max(0, int(limit))]
        log.info("limit=%d, converting %d root(s)", int(limit), len(roots))
    if dry_run:
        for path in roots:
            log.info("  %s", _rel_source(source_root, path))
        return output_root

    if prompt is not None:
        log.info("task prompt: %s (all episodes)", prompt)
    elif from_source_task:
        log.info("task prompt: copy from source LeRobot tasks")
    else:
        log.info("task prompt: placeholder %s (all episodes)", PLACEHOLDER_TASK)
    meta_task_prompt = prompt if prompt is not None else PLACEHOLDER_TASK

    if output_root.exists():
        if resume and not overwrite:
            pass
        elif overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"{output_root} exists; pass --overwrite or --resume")

    cfg = EEActionConfig(max_interp_gap=max_interp_gap)
    sources_path = output_root / "meta" / "episode_sources.jsonl"
    skipped_path = output_root / "meta" / "skipped.jsonl"
    out_episodes: list[dict[str, Any]] = []
    out_stats: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    total_frames = 0
    video_hw: tuple[int, int] | None = None
    fps = 30.0
    missing_videos = 0
    skipped = 0
    done_sources: set[str] = set()

    if resume and sources_path.exists():
        for row in _load_jsonl(sources_path):
            done_sources.add(str(row.get("source") or ""))
            out_episodes.append({
                "episode_index": int(row["episode_index"]),
                "tasks": row.get("tasks") or [PLACEHOLDER_TASK],
                "length": int(row.get("length") or 0),
            })
            total_frames += int(row.get("length") or 0)
            for text in out_episodes[-1]["tasks"]:
                task_to_index.setdefault(str(text), len(task_to_index))
        stats_rows = _load_jsonl(output_root / "meta" / "episodes_stats.jsonl")
        out_stats = [row for row in stats_rows if "episode_index" in row]
        info_path = output_root / "meta" / "info.json"
        if info_path.exists():
            prev = _read_json(info_path)
            fps = float(prev.get("fps") or fps)
            feat = ((prev.get("features") or {}).get(video_key) or {})
            shape = feat.get("shape") or [256, 456, 3]
            video_hw = (int(shape[0]), int(shape[1])) if len(shape) >= 2 else video_hw
        log.info("resume: %d episodes already written, skip %d sources", len(out_episodes), len(done_sources))

    started = time.time()
    for src_i, clip in enumerate(roots):
        rel = _rel_source(source_root, clip)
        if rel in done_sources:
            continue
        try:
            info = _load_info(clip)
            clip_fps = float(info.get("fps") or 30.0)
            if out_episodes and abs(clip_fps - fps) > 1e-3:
                log.warning("%s fps=%.3f differs from dataset fps=%.3f", rel, clip_fps, fps)
            else:
                fps = clip_fps
            episodes = _load_episode_meta(clip, info, video_key)
            task_map = _load_task_map(clip) if from_source_task and prompt is None else {}
            cache: dict[Path, pd.DataFrame] = {}
            for episode in episodes:
                arrays = _episode_ee_arrays(_resolve_data_table(clip, info, episode, cache))
                ee = derive_bimanual_ee(
                    left_xyz=arrays["left_xyz"],
                    right_xyz=arrays["right_xyz"],
                    left_rot=arrays["left_rot"],
                    right_rot=arrays["right_rot"],
                    cam_R=arrays["cam_R"],
                    cam_t=arrays["cam_t"],
                    left_valid=arrays["left_valid"],
                    right_valid=arrays["right_valid"],
                    left_kp21=arrays["left_kp21"],
                    right_kp21=arrays["right_kp21"],
                    config=cfg,
                )
                state = ee["state"]
                action = ee["action"] if action_mode == "relative" else derive_absolute_action(state, ee["hand_valid"])
                length = int(len(state))
                ep_index = len(out_episodes)
                chunk = _chunk_index(ep_index)
                data_dir = output_root / "data" / f"chunk-{chunk:03d}"
                video_dir = output_root / "videos" / f"chunk-{chunk:03d}" / video_key
                data_dir.mkdir(parents=True, exist_ok=True)
                video_dir.mkdir(parents=True, exist_ok=True)

                task_text = _episode_task_text(
                    prompt=prompt,
                    episode=episode,
                    task_map=task_map,
                    arrays=arrays,
                    from_source_task=from_source_task,
                )
                task_index = task_to_index.setdefault(task_text, len(task_to_index))
                timestamps = arrays["timestamps"]
                if timestamps is None or len(timestamps) != length:
                    timestamps = np.arange(length, dtype=np.float64) / fps if fps > 0 else np.zeros(length)

                src_video = _resolve_video(clip, info, episode, video_key)
                dst_video = video_dir / f"episode_{ep_index:06d}.mp4"
                if src_video.exists():
                    _link_or_copy(src_video, dst_video)
                    height, width, n_frames, _codec = _probe_video(dst_video)
                    if video_hw is None:
                        video_hw = (height, width)
                    elif video_hw != (height, width):
                        log.warning("%s video %dx%d vs dataset %dx%d", rel, width, height, video_hw[1], video_hw[0])
                    if abs(n_frames - length) > 2:
                        log.warning("%s frames=%d episode length=%d", dst_video.name, n_frames, length)
                else:
                    missing_videos += 1
                    log.warning("missing source video for %s episode %s: %s", rel, episode.get("episode_index"), src_video)

                table = pa.table(
                    {
                        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), STATE_DIM)),
                        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)),
                        "observation.hand_valid": pa.array(ee["hand_valid"].astype(bool).tolist(), type=pa.list_(pa.bool_(), 2)),
                        "episode_index": np.full(length, ep_index, dtype=np.int64),
                        "frame_index": np.arange(length, dtype=np.int64),
                        "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                        "task_index": np.full(length, task_index, dtype=np.int64),
                        "timestamp": timestamps.astype(np.float32),
                    }
                )
                pq.write_table(table, data_dir / f"episode_{ep_index:06d}.parquet")

                ep_row = {"episode_index": ep_index, "tasks": [task_text], "length": length}
                out_episodes.append(ep_row)
                out_stats.append({
                    "episode_index": ep_index,
                    "stats": {
                        video_key: {
                            "min": [[[0.0]], [[0.0]], [[0.0]]],
                            "max": [[[0.0]], [[0.0]], [[0.0]]],
                            "mean": [[[0.0]], [[0.0]], [[0.0]]],
                            "std": [[[0.0]], [[0.0]], [[0.0]]],
                            "count": [0],
                        },
                        "observation.state": _vector_stats(state),
                        "action": _vector_stats(action),
                        "observation.hand_valid": _vector_stats(ee["hand_valid"].astype(np.float64)),
                        "episode_index": _scalar_stats(np.full(length, ep_index)),
                        "frame_index": _scalar_stats(np.arange(length)),
                        "index": _scalar_stats(np.arange(total_frames, total_frames + length)),
                        "task_index": _scalar_stats(np.full(length, task_index)),
                        "timestamp": _scalar_stats(timestamps),
                    },
                })
                _append_jsonl(sources_path, {
                    "episode_index": ep_index,
                    "source": rel,
                    "source_episode_index": int(episode["episode_index"]),
                    "tasks": [task_text],
                    "length": length,
                })
                total_frames += length
        except Exception as exc:
            skipped += 1
            log.exception("skip %s: %s", rel, exc)
            _append_jsonl(skipped_path, {"source": rel, "error": f"{type(exc).__name__}: {exc}"})

        if (src_i + 1) % 20 == 0 or src_i + 1 == len(roots):
            elapsed = max(time.time() - started, 1e-6)
            log.info(
                "progress %d/%d clips  episodes=%d frames=%d skipped=%d  %.1f clip/s",
                src_i + 1, len(roots), len(out_episodes), total_frames, skipped,
                (src_i + 1) / elapsed,
            )
            if video_hw is not None:
                _write_dataset_meta(
                    output_root,
                    out_episodes=out_episodes,
                    out_stats=out_stats,
                    task_to_index=task_to_index,
                    video_key=video_key,
                    video_hw=video_hw,
                    fps=fps,
                    action_mode=action_mode,
                    repo_id=repo_id,
                    total_frames=total_frames,
                    task_prompt=meta_task_prompt,
                )

    if not out_episodes:
        raise RuntimeError(f"no episodes converted from {source_root}")
    if video_hw is None:
        video_hw = (256, 456)
    _write_dataset_meta(
        output_root,
        out_episodes=out_episodes,
        out_stats=out_stats,
        task_to_index=task_to_index,
        video_key=video_key,
        video_hw=video_hw,
        fps=fps,
        action_mode=action_mode,
        repo_id=repo_id,
        total_frames=total_frames,
        task_prompt=meta_task_prompt,
    )
    log.info(
        "wrote %d episodes / %d frames to %s (skipped=%d missing videos=%d action_mode=%s)",
        len(out_episodes), total_frames, output_root, skipped, missing_videos, action_mode,
    )
    return output_root


def _self_test() -> None:
    def rz(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    result = derive_bimanual_ee(
        left_xyz=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        right_xyz=np.zeros((2, 3)),
        left_rot=np.tile(np.eye(3), (2, 1, 1)),
        right_rot=np.tile(np.eye(3), (2, 1, 1)),
        cam_R=np.stack([rz(np.pi / 2), np.eye(3)]),
        cam_t=np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]),
        left_valid=np.array([True, True]),
        right_valid=np.array([False, False]),
    )
    np.testing.assert_allclose(result["state"][0, :3], [1.0, 3.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(result["state"][1, :3], [1.0, 4.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(result["action"][0, :3], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(result["hand_valid"][:, 0], [True, True])
    np.testing.assert_array_equal(result["hand_valid"][:, 1], [False, False])

    joints = np.zeros((2, 21, 3))
    joints[:, 17, 0] = 1.0
    joints[0, 8, 1] = 0.25
    joints[1, 8, 1] = 0.75
    result = derive_bimanual_ee(
        left_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        right_xyz=np.zeros((2, 3)),
        left_rot=np.stack([np.eye(3), rz(np.pi / 2)]),
        right_rot=np.tile(np.eye(3), (2, 1, 1)),
        cam_R=np.eye(3),
        cam_t=np.zeros(3),
        left_valid=np.array([True, True]),
        right_valid=np.array([False, False]),
        left_kp21=joints,
    )
    np.testing.assert_allclose(result["action"][0, :3], [1.0, 2.0, 3.0], atol=1e-6)
    rel = _MANO_TO_EE_LEFT.T @ rz(np.pi / 2) @ _MANO_TO_EE_LEFT
    np.testing.assert_allclose(result["action"][0, 3:6], _rotvec(rel), atol=1e-6)
    left_ee = _mano_to_ee(np.eye(3), is_right=False)
    np.testing.assert_allclose(result["state"][0, 3:6], _rotvec(left_ee), atol=1e-6)
    np.testing.assert_allclose(left_ee[:, 2], [1.0, 0.0, 0.0], atol=1e-6)  # Z = fingers = MANO X
    np.testing.assert_allclose(left_ee[:, 0], [0.0, -1.0, 0.0], atol=1e-6)  # X = palm-out = -MANO Y
    np.testing.assert_allclose(result["state"][0, 6], 0.25, atol=1e-6)
    np.testing.assert_allclose(result["action"][0, 6], 0.75, atol=1e-6)
    print("[OK] self-test")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="One lerobot_v3 root, or a tree of them (e.g. Egocentric-100K)")
    parser.add_argument("--output", default=None, help="OpenPI LeRobot v2.1 output root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep existing output and skip already converted clips")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many lerobot_v3 roots")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered lerobot_v3 roots")
    parser.add_argument("--video-key", default=DEFAULT_VIDEO_KEY)
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="relative: ego2robot / pi05_ego_bimanual; absolute: store next-frame EE like 0908_video",
    )
    parser.add_argument(
        "--prompt",
        "--task-prompt",
        dest="prompt",
        default=None,
        help="If set, every episode uses this task prompt. Otherwise use the placeholder "
             f"{PLACEHOLDER_TASK!r}.",
    )
    parser.add_argument(
        "--from-source-task",
        action="store_true",
        help="Copy existing LeRobot task text when --prompt is omitted. Ignored if --prompt is set.",
    )
    parser.add_argument("--max-interp-gap", type=int, default=3)
    parser.add_argument("--repo-id", default="local/ego-pi05-bimanual")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.self_test:
        _self_test()
        if not args.source:
            return 0
    if not args.source:
        parser.error("--source is required unless --self-test")
    out = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT
    convert(
        Path(args.source),
        out,
        overwrite=args.overwrite,
        video_key=args.video_key,
        action_mode=args.action_mode,
        prompt=args.prompt,
        from_source_task=args.from_source_task,
        max_interp_gap=args.max_interp_gap,
        repo_id=args.repo_id,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(f"[OK] output: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
