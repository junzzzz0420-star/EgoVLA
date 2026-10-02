# -*- coding: utf-8 -*-
"""DAS double-sphere 鱼眼 → pinhole 去畸变。

公式来源: Usenko et al., "The Double Sphere Camera Model", 3DV 2018 (arXiv:1807.08957) Eq.40-51。
DAS 的 camera_info 里 D = [fx, fy, cx, cy, xi, alpha]。

用法:
    # 自检 DS 公式 round-trip（应 < 1e-9 rad）
    python undistort_ds.py --calib calib.json --selftest

    # 只矩形化一帧存 PNG（快速目检直线/构图）
    python undistort_ds.py --calib calib.json --video camera0.mp4 \
        --frame-only 300 --png /tmp/frame300.png

    # 整段视频去畸变
    python undistort_ds.py --calib calib.json --video camera0.mp4 \
        --out camera0_rect.mp4 --hfov 120 --width 1536 --height 864

输出目标相机为针孔模型；--pitch-deg>0 表示目标光轴相对源光轴向下俯（度），
--yaw-deg>0 向右。矩形化后的新内参写成 <out>.json（或 --png 时打印）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- DS 模型
def ds_project(P: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               xi: float, alpha: float):
    """3D 点 (...,3) → 像素 (...,2)。无效点返回 NaN。"""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    d2 = np.sqrt(x * x + y * y + (xi * d1 + z) ** 2)
    denom = alpha * d2 + (1.0 - alpha) * (xi * d1 + z)
    ok = denom > EPS
    denom = np.where(ok, denom, 1.0)
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    return np.stack([u, v], axis=-1), ok


def ds_unproject(uv: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float):
    """像素 (...,2) → 单位射线 (...,3)。无效像素返回 NaN。"""
    mx = (uv[..., 0] - cx) / fx
    my = (uv[..., 1] - cy) / fy
    r2 = mx * mx + my * my
    root = 1.0 - (2.0 * alpha - 1.0) * r2
    ok = root >= 0.0
    root = np.sqrt(np.where(ok, root, 0.0))
    mz = (1.0 - alpha * alpha * r2) / (alpha * root + (1.0 - alpha))
    ok &= mz > 0.0
    sqrt1 = np.sqrt(mz * mz + (1.0 - xi * xi) * r2)
    k = (mz * xi + sqrt1) / (mz * mz + r2)
    X = np.stack([k * mx, k * my, k * mz - xi], axis=-1)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    X = X / np.where(n > 0, n, 1.0)
    return np.where(ok[..., None], X, np.nan)


def rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- 工具
def load_calib(calib_path: str) -> dict:
    with open(calib_path, encoding="utf-8") as f:
        meta = json.load(f)
    ds = meta["calibration"]["ds_params"]
    return {
        "fx": float(ds["fx"]), "fy": float(ds["fy"]),
        "cx": float(ds["cx"]), "cy": float(ds["cy"]),
        "xi": float(ds["xi"]), "alpha": float(ds["alpha"]),
        "width": int(meta["calibration"]["width"]),
        "height": int(meta["calibration"]["height"]),
        "meta": meta,
    }


def source_fov_deg(cal: dict) -> dict:
    """图像边缘/角落对应的入射角（从光轴量起）。"""
    w, h = cal["width"], cal["height"]
    pts = {
        "center": (cal["cx"], cal["cy"]),
        "left_edge": (0, cal["cy"]), "right_edge": (w - 1, cal["cy"]),
        "top_edge": (cal["cx"], 0), "bottom_edge": (cal["cx"], h - 1),
        "corner_bl": (0, h - 1), "corner_br": (w - 1, h - 1),
        "corner_tl": (0, 0), "corner_tr": (w - 1, 0),
    }
    out = {}
    for name, (u, v) in pts.items():
        ray = ds_unproject(np.array([[u, v]], dtype=np.float64),
                           cal["fx"], cal["fy"], cal["cx"], cal["cy"],
                           cal["xi"], cal["alpha"])[0]
        if np.isnan(ray).any():
            out[name] = None
        else:
            out[name] = float(np.degrees(np.arccos(np.clip(ray[2], -1, 1))))
    return out


def build_remap(cal: dict, out_w: int, out_h: int, hfov_deg: float,
                pitch_deg: float, yaw_deg: float, roll_deg: float):
    """目标 pinhole 像素 → 源 DS 像素的重映射表。"""
    f = (out_w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    cx_t, cy_t = (out_w - 1) / 2.0, (out_h - 1) / 2.0

    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float64),
                         np.arange(out_h, dtype=np.float64))
    P = np.stack([(uu - cx_t) / f, (vv - cy_t) / f, np.ones_like(uu)], axis=-1)

    # pitch>0 向下俯: 光学系 y 向下, 绕 x 轴 -pitch 把 z 转向 +y
    R = rot_z(roll_deg) @ rot_y(yaw_deg) @ rot_x(-pitch_deg)
    P = P @ R.T

    uv_src, ok = ds_project(P, cal["fx"], cal["fy"], cal["cx"], cal["cy"],
                            cal["xi"], cal["alpha"])
    inb = ok & (uv_src[..., 0] >= 0) & (uv_src[..., 0] <= cal["width"] - 1) \
            & (uv_src[..., 1] >= 0) & (uv_src[..., 1] <= cal["height"] - 1)
    map_x = np.where(inb, uv_src[..., 0], -1).astype(np.float32)
    map_y = np.where(inb, uv_src[..., 1], -1).astype(np.float32)
    new_K = [[f, 0.0, cx_t], [0.0, f, cy_t], [0.0, 0.0, 1.0]]
    return map_x, map_y, new_K, float(1.0 - inb.mean())


# ---------------------------------------------------------------- 主流程
def selftest(cal: dict) -> None:
    rng = np.random.default_rng(0)
    # ±80° 锥内随机射线（避开模型边界）
    n = 200000
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2]) + 1.5
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    uv, ok = ds_project(v, cal["fx"], cal["fy"], cal["cx"], cal["cy"],
                        cal["xi"], cal["alpha"])
    rays = ds_unproject(uv, cal["fx"], cal["fy"], cal["cx"], cal["cy"],
                        cal["xi"], cal["alpha"])
    good = ok & ~np.isnan(rays).any(axis=1)
    ang = np.degrees(np.arccos(np.clip(
        (v[good] * rays[good]).sum(axis=1), -1, 1)))
    print(f"[selftest] rays={n} valid={good.sum()}  "
          f"max angular err={ang.max():.3e} deg  mean={ang.mean():.3e} deg")
    fov = source_fov_deg(cal)
    print("[selftest] source FOV (入射角, 度):")
    for k, val in fov.items():
        print(f"    {k:<12} {val if val is None else round(val, 2)}")
    if not good.all() or ang.max() > 1e-4:
        print("[selftest] FAILED", file=sys.stderr)
        raise SystemExit(1)


def rectify_frame(frame_bgr: np.ndarray, map_x, map_y):
    return cv2.remap(frame_bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def run_video(args) -> None:
    cal = load_calib(args.calib)
    map_x, map_y, new_K, invalid_ratio = build_remap(
        cal, args.width, args.height, args.hfov,
        args.pitch_deg, args.yaw_deg, args.roll_deg)
    print(f"[rect] {args.width}x{args.height} hfov={args.hfov}° "
          f"pitch={args.pitch_deg}° yaw={args.yaw_deg}° roll={args.roll_deg}° "
          f"invalid={invalid_ratio * 100:.2f}%")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        raise SystemExit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.frame_only is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_only)
        ok, frame = cap.read()
        if not ok:
            print(f"cannot read frame {args.frame_only}", file=sys.stderr)
            raise SystemExit(1)
        out_png = Path(args.png) if args.png else Path(
            f"{Path(args.video).stem}_frame{args.frame_only}_rect.png")
        cv2.imwrite(str(out_png), rectify_frame(frame, map_x, map_y))
        print(f"[rect] wrote {out_png}")
        print(f"[rect] new K = {np.round(np.array(new_K), 2).tolist()}")
        cap.release()
        return

    if not args.out:
        print("need --out for full video", file=sys.stderr)
        raise SystemExit(1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{args.width}x{args.height}", "-r", f"{src_fps:.6f}",
           "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(rectify_frame(frame, map_x, map_y).tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    cap.release()
    if proc.returncode != 0:
        print("ffmpeg failed", file=sys.stderr)
        raise SystemExit(1)

    meta = {
        "source_video": str(Path(args.video).resolve()),
        "source_calib": str(Path(args.calib).resolve()),
        "model": "pinhole",
        "width": args.width, "height": args.height,
        "K": new_K,
        "hfov_deg": args.hfov,
        "pitch_deg": args.pitch_deg, "yaw_deg": args.yaw_deg,
        "roll_deg": args.roll_deg,
        "R_src_from_tgt": (rot_z(args.roll_deg) @ rot_y(args.yaw_deg)
                           @ rot_x(-args.pitch_deg)).tolist(),
        "frames": n, "fps": src_fps,
        "invalid_pixel_ratio": invalid_ratio,
    }
    with open(str(out_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[rect] wrote {out_path} ({n} frames) + {out_path}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, help="export_mcap.py 写出的 calib.json")
    ap.add_argument("--video", help="camera0.mp4")
    ap.add_argument("--out", help="去畸变视频输出路径(.mp4)")
    ap.add_argument("--png", help="--frame-only 模式的 PNG 输出")
    ap.add_argument("--frame-only", type=int, default=None, help="只处理第 N 帧")
    ap.add_argument("--hfov", type=float, default=120.0, help="目标水平 FOV(度)")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--pitch-deg", type=float, default=0.0, help=">0 目标光轴向下")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help=">0 目标光轴向右")
    ap.add_argument("--roll-deg", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cal = load_calib(args.calib)
    if args.selftest:
        selftest(cal)
        return
    if not args.video:
        ap.error("need --video (or --selftest)")
    run_video(args)


if __name__ == "__main__":
    main()
