"""Side-by-side visualization of Pi0.5 14-D EE state/action vs ego video.

Left panel: ego video with both end-effectors projected (OpenCV camera frame).
Right panel: third-person 3D of the stored 14-D state, action arrows, and
the pose reconstructed by integrating relative actions.

Example::

    python ego_pipeline/tasks/visualize_pi05_ee.py \\
        --dataset /data/dataset/pretrain_data/ego/lerobot_pi05 \\
        --source-v3 /path/to/lerobot_v3 \\
        --output /tmp/pi05_ee_viz.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as _R

log = logging.getLogger("ego_pipeline.visualize_pi05_ee")

VIDEO_KEY = "observation.images.ego"
Z_UP_FROM_CV = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
LEFT_BGR = (255, 210, 77)
RIGHT_BGR = (77, 211, 255)
RECON_BGR = (180, 180, 180)
AXIS_BGR = ((80, 80, 255), (80, 220, 80), (255, 160, 60))
BG = np.array([18, 28, 32], dtype=np.uint8)


def _rot(vec: np.ndarray) -> np.ndarray:
    return _R.from_rotvec(np.asarray(vec, dtype=np.float64).reshape(3)).as_matrix()


def _split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return state[:, :3], state[:, 3:6], state[:, 7:10], state[:, 10:13]


def _integrate_relative(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    recon = np.zeros_like(state, dtype=np.float64)
    recon[0] = state[0]
    for t in range(len(state) - 1):
        for base in (0, 7):
            r_t = _rot(recon[t, base + 3:base + 6])
            recon[t + 1, base:base + 3] = recon[t, base:base + 3] + action[t, base:base + 3]
            recon[t + 1, base + 3:base + 6] = _R.from_matrix(
                r_t @ _rot(action[t, base + 3:base + 6])
            ).as_rotvec()
            recon[t + 1, base + 6] = action[t, base + 6]
    return recon.astype(np.float32)


def _load_episode(root: Path, episode: int, video_key: str = VIDEO_KEY) -> dict[str, Any]:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 30.0)
    parquet = root / f"data/chunk-000/episode_{episode:06d}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    df = pq.read_table(parquet).to_pandas()
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    valid = np.stack(df["observation.hand_valid"].to_numpy()).astype(bool)
    video = root / f"videos/chunk-000/{video_key}/episode_{episode:06d}.mp4"
    frames = _read_video(video)
    return {
        "info": info,
        "fps": fps,
        "state": state,
        "action": action,
        "valid": valid,
        "frames": frames,
        "video": video,
    }


def _read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    return frames


def _load_source_camera(source_v3: Path | None, frames: int) -> dict[str, np.ndarray] | None:
    if source_v3 is None:
        return None
    root = Path(source_v3)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        log.warning("no parquet under %s", root)
        return None
    df = pq.read_table(paths[0]).to_pandas()
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == int(df["episode_index"].iloc[0])]
    df = df.sort_values("frame_index") if "frame_index" in df.columns else df
    df = df.iloc[:frames]
    K = np.asarray(df["intrinsics"].iloc[0], dtype=np.float64).reshape(3, 3)
    w2c = np.stack([np.asarray(x, dtype=np.float64).reshape(4, 4) for x in df["extrinsics_w2c"]])
    return {"K": K, "R": w2c[:, :3, :3], "t": w2c[:, :3, 3]}


def _estimate_K(height: int, width: int) -> np.ndarray:
    fx = 0.9 * max(height, width)
    return np.array([[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]])


def _to_current_camera(
    xyz: np.ndarray,
    rotvec: np.ndarray,
    cam: dict[str, np.ndarray] | None,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    r_g = _rot(rotvec)
    if cam is None:
        return xyz, r_g
    r0, t0 = cam["R"][0], cam["t"][0]
    rt, tt = cam["R"][frame], cam["t"][frame]
    xyz_world = r0.T @ (xyz - t0)
    xyz_cam = rt @ xyz_world + tt
    rot_cam = rt @ r0.T @ r_g
    return xyz_cam, rot_cam


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uv = (K @ pts.T).T
    z = uv[:, 2:3]
    uv = uv[:, :2] / np.clip(z, 1e-6, None)
    uv[z[:, 0] <= 1e-6] = np.nan
    return uv


def _uv(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    return _project(np.asarray(points, dtype=np.float64), K)


def _poly(image: np.ndarray, pts_uv: np.ndarray, color, fill: bool, thickness: int = 2) -> None:
    if not np.isfinite(pts_uv).all():
        return
    contour = np.rint(pts_uv).astype(np.int32).reshape(-1, 1, 2)
    if fill:
        overlay = image.copy()
        cv2.fillConvexPoly(overlay, contour, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)


def _gripper_geometry(p: np.ndarray, rot: np.ndarray, grip: float, scale: float = 1.0) -> dict[str, Any]:
    """Build a readable parallel-jaw hand in the EE frame.

    Local axes come from the stored rotvec: X = palm normal, Y = gripper
    opening, Z = finger forward.
    """
    p = np.asarray(p, dtype=np.float64).reshape(3)
    x, y, z = rot[:, 0], rot[:, 1], rot[:, 2]
    g = float(np.clip(grip, 0.0, 1.0))
    s = float(scale)
    palm_w, palm_l = 0.08 * s, 0.06 * s
    c = p + z * (palm_l * 0.15)
    palm = np.stack([
        c + 0.5 * palm_w * y - 0.5 * palm_l * z - 0.004 * s * x,
        c - 0.5 * palm_w * y - 0.5 * palm_l * z - 0.004 * s * x,
        c - 0.5 * palm_w * y + 0.5 * palm_l * z - 0.004 * s * x,
        c + 0.5 * palm_w * y + 0.5 * palm_l * z - 0.004 * s * x,
    ])
    half = (0.008 + 0.048 * g) * s
    jaw_l, jaw_t = 0.07 * s, 0.016 * s

    def jaw(sign: float) -> np.ndarray:
        o = p + sign * half * y + 0.02 * s * z
        yy = sign * y
        return np.stack([
            o + 0.55 * jaw_t * yy - 0.008 * s * x,
            o - 0.20 * jaw_t * yy - 0.008 * s * x,
            o - 0.20 * jaw_t * yy - 0.008 * s * x + jaw_l * z,
            o + 0.55 * jaw_t * yy - 0.008 * s * x + jaw_l * z,
        ])

    ax = 0.13 * s
    return {
        "palm": palm,
        "jaw_pos": jaw(1.0),
        "jaw_neg": jaw(-1.0),
        "normal": (p, p + ax * x),
        "opening": (p - half * y, p + half * y),
        "forward": (p, p + 1.15 * ax * z),
        "axes": ((p, p + ax * x), (p, p + ax * y), (p, p + 1.15 * ax * z)),
        "label": p - 0.03 * s * x + 0.02 * s * z,
        "grip": g,
        "width_m": 2.0 * half / s,
        "half": half,
    }


def _draw_ee_overlay(
    image: np.ndarray,
    xyz: np.ndarray,
    rotvec: np.ndarray,
    grip: float,
    valid: bool,
    color: tuple[int, int, int],
    K: np.ndarray,
    cam: dict[str, np.ndarray] | None,
    frame: int,
    action_xyz: np.ndarray | None = None,
    tag: str = "",
) -> None:
    if not valid:
        return
    p, r = _to_current_camera(xyz, rotvec, cam, frame)
    geom = _gripper_geometry(p, r, grip)
    _poly(image, _uv(geom["palm"], K), color, True, 2)
    _poly(image, _uv(geom["jaw_pos"], K), color, True, 2)
    _poly(image, _uv(geom["jaw_neg"], K), color, True, 2)
    origin = _uv(p, K)[0]
    if not np.isfinite(origin).all():
        return
    o = tuple(np.rint(origin).astype(int))
    labels = ("X", "Y", "Z")
    for (start, end), bgr, name in zip(geom["axes"], AXIS_BGR, labels):
        uv = _uv(np.stack([start, end]), K)
        if np.isfinite(uv).all():
            a, b = tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int))
            cv2.arrowedLine(image, a, b, bgr, 2, cv2.LINE_AA, tipLength=0.18)
            cv2.putText(image, name, (b[0] + 2, b[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    cv2.circle(image, o, 6, color, -1, cv2.LINE_AA)
    if tag:
        cv2.putText(
            image, f"{tag} g={grip:.2f}", (o[0] + 8, o[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    if action_xyz is not None and np.linalg.norm(action_xyz) > 1e-4:
        a_uv = _uv(np.stack([p, p + np.asarray(action_xyz) * 8.0]), K)
        if np.isfinite(a_uv).all():
            cv2.arrowedLine(
                image,
                tuple(np.rint(a_uv[0]).astype(int)),
                tuple(np.rint(a_uv[1]).astype(int)),
                color, 2, cv2.LINE_AA, tipLength=0.2,
            )


def _orbit_project(points: np.ndarray, width: int, height: int, az: float, el: float, zoom: float):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3) @ Z_UP_FROM_CV.T
    center = np.median(pts, axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    radius = max(radius, 0.12) * 1.08
    scale = min(width, height) * 0.46 * zoom / radius
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)

    def project(point):
        x, y, z = np.asarray(point, dtype=np.float64) @ Z_UP_FROM_CV.T - center
        x1, depth = x * ca + y * sa, -x * sa + y * ca
        y2 = -z * ce - depth * se
        return np.array([width * 0.5 + x1 * scale, height * 0.52 + y2 * scale])

    def depth_of(point):
        x, y, z = np.asarray(point, dtype=np.float64) @ Z_UP_FROM_CV.T - center
        depth = -x * sa + y * ca
        return float(depth * ce - z * se)

    return project, depth_of, radius, center, scale


def _line(frame, a, b, color, thickness=2):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return
    cv2.line(frame, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, thickness, cv2.LINE_AA)


def _fill_poly_3d(panel, project, pts3, color, thickness: int = 2) -> None:
    uv = np.stack([project(p) for p in pts3])
    _poly(panel, uv, color, True, thickness)


def _draw_grip_bar(panel, x: int, y: int, w: int, h: int, value: float, color, label: str) -> None:
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(panel, (x, y), (x + w, y + h), (40, 48, 52), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), color, 1)
    fill = max(1, int(round(w * value)))
    cv2.rectangle(panel, (x + 1, y + 1), (x + fill - 1, y + h - 1), color, -1)
    state = "open" if value > 0.55 else "mid" if value > 0.25 else "closed"
    cv2.putText(
        panel, f"{label}  {value:.2f}  {state}",
        (x + w + 10, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
    )


def _draw_hand_3d(panel, project, p, rotv, grip, color, tag: str, act) -> None:
    r = _rot(rotv)
    geom = _gripper_geometry(p, r, grip, scale=1.85)
    _fill_poly_3d(panel, project, geom["palm"], color, 3)
    jaw = tuple(min(255, int(c * 1.12)) for c in color)
    _fill_poly_3d(panel, project, geom["jaw_pos"], jaw, 3)
    _fill_poly_3d(panel, project, geom["jaw_neg"], jaw, 3)
    _line(panel, project(geom["opening"][0]), project(geom["opening"][1]), (245, 245, 245), 3)
    labels = ("X nml", "Y open", "Z fwd")
    offsets = (np.array([6, -8]), np.array([6, 16]), np.array([6, -8]))
    for (start, end), bgr, name, off in zip(geom["axes"], AXIS_BGR, labels, offsets):
        a, b = project(start), project(end)
        _line(panel, a, b, bgr, 4)
        if np.isfinite(b).all():
            cv2.putText(
                panel, name, tuple(np.rint(b + off).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, bgr, 1, cv2.LINE_AA,
            )
    o = project(p)
    if np.isfinite(o).all():
        oi = tuple(np.rint(o).astype(int))
        cv2.circle(panel, oi, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, oi, 6, color, -1, cv2.LINE_AA)
        dy = -18 if tag == "L" else 22
        cv2.putText(
            panel, f"{tag}  g={grip:.2f}  {geom['width_m']*100:.1f}cm",
            (oi[0] + 12, oi[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA,
        )
    if act is not None and np.linalg.norm(act) > 1e-4:
        _line(panel, project(p), project(p + np.asarray(act, dtype=np.float64) * 10.0), color, 2)


def _draw_3d(
    height: int,
    width: int,
    state: np.ndarray,
    recon: np.ndarray,
    action: np.ndarray,
    valid: np.ndarray,
    frame: int,
) -> np.ndarray:
    panel = np.broadcast_to(BG, (height, width, 3)).copy()
    lxyz, _, rxyz, _ = _split_state(state)
    hands = []
    if valid[:, 0].any():
        hands.append(lxyz[valid[:, 0]])
    if valid[:, 1].any():
        hands.append(rxyz[valid[:, 1]])
    cloud = np.concatenate(hands) if hands else np.zeros((1, 3))
    glyph = 0.22
    padded = np.vstack([cloud, cloud + glyph, cloud - glyph])
    project, depth_of, radius, _, scale = _orbit_project(
        padded, width, height, az=-0.55, el=0.42, zoom=1.35,
    )

    half = max(0.18, radius)
    step = 0.05
    y_ground = float(np.nanmax(cloud[:, 1]) + 0.05) if cloud.size else 0.3
    cx, cz = float(np.median(cloud[:, 0])), float(np.median(cloud[:, 2]))
    for x in np.arange(cx - half, cx + half + 1e-6, step):
        _line(panel, project([x, y_ground, cz - half]), project([x, y_ground, cz + half]), (50, 68, 74), 1)
    for z in np.arange(cz - half, cz + half + 1e-6, step):
        _line(panel, project([cx - half, y_ground, z]), project([cx + half, y_ground, z]), (50, 68, 74), 1)

    def trail(xyz, mask, color):
        pts = [project(p) for p, ok in zip(xyz[: frame + 1], mask[: frame + 1]) if ok]
        for a, b in zip(pts[:-1], pts[1:]):
            _line(panel, a, b, color, 3)

    trail(lxyz, valid[:, 0], LEFT_BGR)
    trail(rxyz, valid[:, 1], RIGHT_BGR)

    items = []
    if valid[frame, 0]:
        items.append((depth_of(state[frame, :3]), "L", state[frame, :3], state[frame, 3:6], state[frame, 6], LEFT_BGR, action[frame, :3]))
    if valid[frame, 1]:
        items.append((depth_of(state[frame, 7:10]), "R", state[frame, 7:10], state[frame, 10:13], state[frame, 13], RIGHT_BGR, action[frame, 7:10]))
    for _depth, tag, xyz, rotv, grip, color, act in sorted(items, key=lambda item: item[0], reverse=True):
        _draw_hand_3d(panel, project, xyz, rotv, grip, color, tag, act)
        recon_xyz = recon[frame, :3] if tag == "L" else recon[frame, 7:10]
        cv2.circle(panel, tuple(np.rint(project(recon_xyz)).astype(int)), 4, RECON_BGR, 1, cv2.LINE_AA)

    cv2.putText(panel, "3D EE  zoomed to hands", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "X=palm normal   Y=gripper open   Z=finger forward", (18, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 178, 186), 1, cv2.LINE_AA)
    _draw_grip_bar(panel, 18, height - 78, 220, 18, float(state[frame, 6]), LEFT_BGR, "L grip")
    _draw_grip_bar(panel, 18, height - 48, 220, 18, float(state[frame, 13]), RIGHT_BGR, "R grip")
    bar = max(28, int(round(0.10 * scale)))
    x1, y = width - 28, height - 24
    cv2.line(panel, (x1 - bar, y), (x1, y), (230, 230, 230), 3, cv2.LINE_AA)
    cv2.putText(panel, "10 cm", (x1 - bar, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = height / h
    resized = cv2.resize(image, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_LINEAR)
    return resized


def _hud(image: np.ndarray, frame: int, total: int, state: np.ndarray, action: np.ndarray, valid: np.ndarray) -> None:
    cv2.putText(image, f"t={frame:03d}/{total - 1:03d}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    lg, rg = float(state[6]), float(state[13])
    cv2.putText(
        image,
        f"L xyz {state[0]:.2f},{state[1]:.2f},{state[2]:.2f}  valid={int(valid[0])}",
        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LEFT_BGR, 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"R xyz {state[7]:.2f},{state[8]:.2f},{state[9]:.2f}  valid={int(valid[1])}",
        (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RIGHT_BGR, 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"dL {action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}   dR {action[7]:+.3f},{action[8]:+.3f},{action[9]:+.3f}",
        (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA,
    )
    h, _w = image.shape[:2]
    _draw_grip_bar(image, 12, h - 52, 160, 14, lg, LEFT_BGR, "L grip")
    _draw_grip_bar(image, 12, h - 28, 160, 14, rg, RIGHT_BGR, "R grip")


def _encode_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "pipe:0",
        "-frames:v", str(len(frames)),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-g", str(max(1, int(round(fps)))),
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        proc.stdin = None
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def visualize(
    dataset: Path,
    output: Path,
    *,
    episode: int = 0,
    source_v3: Path | None = None,
    snapshot_dir: Path | None = None,
) -> Path:
    data = _load_episode(Path(dataset), episode)
    state, action, valid, frames, fps = data["state"], data["action"], data["valid"], data["frames"], data["fps"]
    n = min(len(state), len(frames))
    recon = _integrate_relative(state[:n], action[:n])
    recon_err = float(np.mean(np.linalg.norm(recon[:, :3] - state[:n, :3], axis=1)))
    log.info("integrated-action vs state mean |dxyz_L|=%.4f m", recon_err)

    cam = _load_source_camera(source_v3, n)
    h0, w0 = frames[0].shape[:2]
    K = cam["K"] if cam is not None else _estimate_K(h0, w0)
    panel_h = 960
    panel_w = 1100
    composed = []
    for t in range(n):
        vis = frames[t].copy()
        _draw_ee_overlay(
            vis, state[t, :3], state[t, 3:6], state[t, 6], bool(valid[t, 0]),
            LEFT_BGR, K, cam, t, action[t, :3], tag="L",
        )
        _draw_ee_overlay(
            vis, state[t, 7:10], state[t, 10:13], state[t, 13], bool(valid[t, 1]),
            RIGHT_BGR, K, cam, t, action[t, 7:10], tag="R",
        )
        _hud(vis, t, n, state[t], action[t], valid[t])
        left = _pad_to_height(vis, panel_h)
        right = _draw_3d(panel_h, panel_w, state[:n], recon, action[:n], valid[:n], t)
        gap = np.full((panel_h, 8, 3), 12, dtype=np.uint8)
        row = np.concatenate([left, gap, right], axis=1)
        h, w = row.shape[:2]
        if h % 2 or w % 2:
            row = cv2.copyMakeBorder(row, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(12, 12, 12))
        composed.append(row)

    output = Path(output)
    _encode_mp4(composed, output, fps)
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        picks = sorted({0, n // 3, 2 * n // 3, n - 1})
        for t in picks:
            cv2.imwrite(str(snapshot_dir / f"frame_{t:03d}.png"), composed[t])
    log.info("wrote %s (%d frames)", output, n)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="lerobot_pi05 root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--source-v3", default=None, help="optional EgoPipeline v3 root for per-frame extrinsics/K")
    parser.add_argument("--snapshots", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    dataset = Path(args.dataset)
    out = Path(args.output) if args.output else dataset / f"viz_episode_{args.episode:06d}.mp4"
    snaps = Path(args.snapshots) if args.snapshots else out.parent / f"{out.stem}_frames"
    visualize(
        dataset,
        out,
        episode=args.episode,
        source_v3=Path(args.source_v3) if args.source_v3 else None,
        snapshot_dir=snaps,
    )
    print(f"[OK] {out.resolve()}")
    print(f"[OK] snapshots {snaps.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
