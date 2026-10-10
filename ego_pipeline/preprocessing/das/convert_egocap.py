# -*- coding: utf-8 -*-
"""EgoCap 原始 bag（mcap, CDR）→ DAS 兼容导出目录（固化转换链，无需 agent）。

输入（同事 LY Z 的数据）:
    /mnt/d/work/ego/LYZ/<录制目录>/bag/bag_X.mcap     （或直接给 mcap 路径/含 mcap 的目录）

输出（--out-root 下，默认 examples/egocap_das_export/<name>/）:
    camera0.mp4 / camera1.mp4          left/right 原始图像重编码（pinhole 直通）
    camera{0,1}_timestamps.npz         log_time_ns（相对，帧时间线）
    imu.npz                            t_log_ns（绝对钟）+ gyro_rad_s + accel_g
    calib_camera{0,1}.json 等          调 make_egocap_calibration 生成 fallback 双目标定
    camera{0,1}_rect.mp4(+json)        直通副本（fuse 读取 K）

用法:
    python ego_pipeline/preprocessing/das/convert_egocap.py \
        --input /mnt/d/work/ego/LYZ/egocap_xxx/bag --name egocap_xxx
转换完成后，界面数据源下拉即出现 "EgoCap已转换 · <name>/"，可直接选运行。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

G = 9.80665
CAM_MAP = {'camera0': '/egocap/camera/left/image_raw',
           'camera1': '/egocap/camera/right/image_raw'}
IMU_TOPIC = '/egocap/imu/data_raw'


def find_mcap(p: Path) -> Path:
    if p.is_file() and p.suffix == '.mcap':
        return p
    for c in [p, p / 'bag']:
        if c.is_dir():
            m = sorted(c.glob('bag_*.mcap')) or sorted(c.glob('*.mcap'))
            if m:
                return m[0]
    raise SystemExit(f'未找到 mcap: {p}')


def default_name(mcap: Path) -> str:
    """egocap_<日期时间>_bag_X：取 mcap 的 egocap* 祖先目录 + 文件名，保证不同批次不重名。"""
    for anc in mcap.parents:
        if anc.name.startswith('egocap'):
            return f'{anc.name}_{mcap.stem}'
    return mcap.stem


def unique_dir(root: Path, name: str) -> Path:
    """同名目录已存在时追加 _1/_2…，永不覆盖旧转换。"""
    d = root / name
    k = 1
    while d.exists():
        d = root / f'{name}_{k}'
        k += 1
    return d


def convert(mcap: Path, out_dir: Path) -> None:
    import cv2
    from mcap.reader import make_reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {}
    stamps = {c: [] for c in CAM_MAP}
    imu_t, gyro, accel = [], [], []
    stamp0 = {}

    f = open(mcap, 'rb')
    reader = make_reader(f)
    try:
        topics = list(CAM_MAP.values()) + [IMU_TOPIC]
        for schema, channel, message in reader.iter_messages():
            if channel.topic not in topics:
                continue
            raw = message.data
            if channel.topic == IMU_TOPIC:
                m = ts.deserialize_cdr(raw, 'sensor_msgs/msg/Imu')
                t = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                imu_t.append(t)
                gyro.append([m.angular_velocity.x, m.angular_velocity.y,
                             m.angular_velocity.z])
                accel.append([m.linear_acceleration.x / G,
                              m.linear_acceleration.y / G,
                              m.linear_acceleration.z / G])       # m/s² → g
            else:
                cam = next(c for c, tp in CAM_MAP.items() if tp == channel.topic)
                m = ts.deserialize_cdr(raw, 'sensor_msgs/msg/Image')
                t = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                stamp0.setdefault(cam, t)
                stamps[cam].append(t - stamp0[cam])                # 相对时间线
                img = np.frombuffer(m.data, np.uint8)
                if m.encoding == 'yuv422_yuy2':
                    # YUYV 4:2:2：每 4 字节 = 2 像素 (Y0 U Y1 V)，手写全范围转换
                    yuv = img.reshape(m.height, m.width // 2, 4).astype(np.float32)
                    y = np.concatenate([yuv[:, :, 0:1], yuv[:, :, 2:3]], axis=2) \
                        .reshape(m.height, m.width)
                    u = np.repeat(yuv[:, :, 1], 2, axis=1)
                    v = np.repeat(yuv[:, :, 3], 2, axis=1)
                    bgr = np.stack([y + 1.772 * (u - 128),
                                    y - 0.344 * (u - 128) - 0.714 * (v - 128),
                                    y + 1.402 * (v - 128)], axis=-1)
                    bgr = np.clip(bgr + 16, 0, 255).astype(np.uint8)  # +16 近似 studio 抬升
                else:
                    g = img.reshape(m.height, m.width, -1)
                    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.shape[2] == 1 \
                        else cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
                if cam not in writers:
                    mp4 = out_dir / f'{cam}.mp4'
                    proc = subprocess.Popen(
                        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                         '-pix_fmt', 'bgr24', '-s', f'{m.width}x{m.height}',
                         '-r', '30', '-i', '-',
                         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                         '-pix_fmt', 'yuv420p', str(mp4)], stdin=subprocess.PIPE)
                    writers[cam] = (proc, m.width, m.height)
                proc, w0, h0 = writers[cam]
                if (m.width, m.height) != (w0, h0):
                    bgr = cv2.resize(bgr, (w0, h0))
                proc.stdin.write(bgr.tobytes())
                print(f'\r  {cam}: {len(stamps[cam])} 帧', end='', flush=True)
    finally:
        try:
            reader.close()
        except AttributeError:
            pass
        f.close()

    print()
    for cam, (proc, *_ ) in writers.items():
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f'{cam}: ffmpeg 编码失败')
        np.savez_compressed(out_dir / f'{cam}_timestamps.npz',
                            log_time_ns=np.asarray(stamps[cam], np.int64),
                            header_stamp_ns=np.asarray(stamps[cam], np.int64),
                            frame_index=np.arange(len(stamps[cam]), dtype=np.int64))
        print(f'[convert] {cam}: {len(stamps[cam])} 帧 → {out_dir / (cam + ".mp4")}')
    np.savez_compressed(out_dir / 'imu.npz',
                        t_log_ns=np.asarray(imu_t, np.int64),
                        gyro_rad_s=np.asarray(gyro, np.float64).reshape(-1, 3),
                        accel_g=np.asarray(accel, np.float64).reshape(-1, 3))
    print(f'[convert] imu: {len(imu_t)} 采样 → {out_dir / "imu.npz"}')

    # fallback 双目标定 + rect 直通副本（fuse 直接可用）
    sys.path.insert(0, str(REPO))
    from make_egocap_calibration import make
    for p in make(out_dir):
        print(f'[convert] {p.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='bag 目录或 bag_X.mcap 路径')
    ap.add_argument('--name', default=None, help='导出名（默认取 mcap 文件名）')
    ap.add_argument('--out-root', default=str(REPO / 'examples' / 'egocap_das_export'))
    args = ap.parse_args()

    mcap = find_mcap(Path(args.input))
    root = Path(args.out_root)
    if args.name:
        name, out = args.name, root / args.name        # 显式命名允许覆盖重转
    else:
        name = default_name(mcap)
        out = unique_dir(root, name)                   # 自动命名永不覆盖
    print(f'[convert] {mcap} → {out}')
    convert(mcap, out)
    print(f'[convert] 导出名: {out.name}')
    print(f'[OK] 界面数据源下拉将出现: EgoCap已转换 · {out.name}/')


if __name__ == '__main__':
    main()
