# -*- coding: utf-8 -*-
"""DAS-Ego mcap → mp4 + IMU npz + 标定 json。

用法:
    python export_mcap.py --input <mcap文件或目录> --camera camera0 \
        --output /path/to/out [--pattern "DAS-Ego"]

每个 mcap 生成:
    <output>/<mcap文件名去扩展名>/camera0.mp4              选中相机的 h264 重封装
    <output>/<...>/camera0_timestamps.npz                  每帧 log_time / header 时间戳(ns)
    <output>/<...>/imu.npz                                 t_log, gyro(N,3)[rad/s], accel(N,3)[g]
    <output>/<...>/calib.json                              DS 内参 + T_b_c + 元数据

h264 流从第一个含 SPS 的消息开始（录制常从会话中途订阅开始，首段缺参数集），
对应被丢弃的帧数记录在 calib.json 的 dropped_frames。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def find_start_codes(buf: bytes):
    """Annex-B 起始码位置（3 字节码是 4 字节码的后缀，扫 3 字节即可覆盖两种）。"""
    pos, out = 0, []
    while True:
        i = buf.find(b"\x00\x00\x01", pos)
        if i < 0:
            return out
        out.append(i + 3)
        pos = i + 3


def nal_has_sps(buf: bytes) -> bool:
    for p in find_start_codes(buf):
        if p < len(buf) and (buf[p] & 0x1F) == 7:
            return True
    return False


def _plain(v):
    """protobuf 标量/repeated 容器 → JSON 可序列化值。"""
    if isinstance(v, (str, bytes, int, float, bool)):
        return v
    try:
        return [float(x) if isinstance(x, (int, float)) else x for x in v]
    except TypeError:
        return v


def vec3(m):
    return None if m is None else (float(m.x), float(m.y), float(m.z))


def scalar_fields(msg) -> dict:
    out = {}
    for fd, v in msg.ListFields():
        if fd.message_type is None and not isinstance(v, (bytes, bytearray)):
            try:
                json.dumps(v)
                out[fd.name] = v
            except (TypeError, ValueError):
                out[fd.name] = str(v)
    return out


def header_stamp_ns(msg) -> float | None:
    h = getattr(msg, "header", None)
    ts = getattr(h, "timestamp", None) if h is not None else None
    return None if ts in (None, 0) else float(ts)


def process_file(mcap_path: Path, out_root: Path, camera: str, prefix: str,
                 keep_stream: bool = False) -> dict:
    cams = ([f'camera{i}' for i in range(6)]
            if camera.lower() in ('all', '6', '*') else camera.split(','))
    return {c: _export_one(mcap_path, out_root, c, prefix, keep_stream) for c in cams}


def _export_one(mcap_path: Path, out_root: Path, camera: str, prefix: str,
                keep_stream: bool) -> dict:
    img_topic = f"{prefix}/sensor/{camera}/compressed"
    info_topic = f"{prefix}/sensor/{camera}/camera_info"
    imu_topic = f"{prefix}/sensor/imu"
    sys_topic = f"{prefix}/system_info"

    frames: list[tuple[float, float | None, bytes]] = []   # log_time, header_ns, data
    calib_fields: dict | None = None
    imu_t: list[float] = []
    imu_g: list[tuple] = []
    imu_a: list[tuple] = []
    imu_broken: str | None = None
    sys_info: dict | None = None

    factory = DecoderFactory()
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        decoders: dict[str, object] = {}
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            need = topic in (img_topic, info_topic, imu_topic, sys_topic)
            if not need:
                continue
            dec = decoders.get(topic)
            if dec is None:
                try:
                    dec = factory.decoder_for(channel.message_encoding, schema)
                except Exception as e:  # 旧固件 schema 可能不是 FDS
                    if topic == imu_topic:
                        imu_broken = str(e)
                        continue
                    if need:
                        return {"file": str(mcap_path),
                                "status": "schema-decode-failed",
                                "topic": topic, "error": str(e)}
                    continue
                decoders[topic] = dec
            decoded = dec(message.data)
            if topic == img_topic:
                frames.append((message.log_time, header_stamp_ns(decoded), bytes(decoded.data)))
            elif topic == info_topic:
                if calib_fields is None:
                    calib_fields = {fd.name: _plain(v)
                                    for fd, v in decoded.ListFields()
                                    if fd.message_type is None}
            elif topic == imu_topic:
                imu_t.append(message.log_time)
                imu_g.append(vec3(decoded.angular_velocity))
                imu_a.append(vec3(decoded.linear_acceleration))
            elif topic == sys_topic and sys_info is None:
                sys_info = scalar_fields(decoded)

    if not frames:
        return {"file": str(mcap_path), "status": "no-camera-messages"}

    # 从第一个含 SPS 的帧开始，保证解码器从干净随机访问点起步
    drop = 0
    for i, (_, _, data) in enumerate(frames):
        if nal_has_sps(data):
            drop = i
            break
    else:
        drop = len(frames)
    if drop:
        frames = frames[drop:]
    if not frames:
        return {"file": str(mcap_path), "status": "no-sps-in-stream"}

    log_t = np.asarray([t for t, _, _ in frames], dtype=np.int64)
    hdr_t = np.asarray([t if t is not None else -1 for _, t, _ in frames], dtype=np.int64)
    clock = hdr_t if (hdr_t > 0).all() else log_t
    dur = (clock[-1] - clock[0]) / 1e9
    fps = (len(frames) - 1) / dur if dur > 0 else 30.0

    out_dir = out_root / mcap_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = out_dir / f"{camera}.h264"
    with open(stream_path, "wb") as w:
        for _, _, data in frames:
            w.write(data)

    mp4_path = out_dir / f"{camera}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "h264",
           "-framerate", f"{fps:.6f}", "-i", str(stream_path),
           "-c", "copy", "-movflags", "+faststart", str(mp4_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"file": str(mcap_path), "status": "ffmpeg-failed",
                "stderr": r.stderr[-2000:]}
    if not keep_stream:
        stream_path.unlink()

    np.savez_compressed(out_dir / f"{camera}_timestamps.npz",
                        log_time_ns=log_t, header_stamp_ns=hdr_t,
                        frame_index=np.arange(len(frames), dtype=np.int64))
    np.savez_compressed(out_dir / "imu.npz",
                        t_log_ns=np.asarray(imu_t, dtype=np.int64),
                        gyro_rad_s=np.asarray(imu_g, dtype=np.float64).reshape(-1, 3),
                        accel_g=np.asarray(imu_a, dtype=np.float64).reshape(-1, 3))

    meta = {
        "source_mcap": str(mcap_path),
        "camera": camera,
        "topic": img_topic,
        "dropped_frames": drop,
        "frames": len(frames),
        "measured_fps": fps,
        "duration_s": dur,
        "t_start_ns": int(clock[0]),
        "t_end_ns": int(clock[-1]),
        "imu_samples": len(imu_t),
        "imu_units": {"gyro": "rad/s", "accel": "g"},
        "imu_schema_error": imu_broken,
        "system_info": sys_info,
    }
    if calib_fields is not None:
        D = list(calib_fields.get("D", []))
        meta["calibration"] = {
            "model": calib_fields.get("distortion_model", "ds"),
            "width": calib_fields.get("width"),
            "height": calib_fields.get("height"),
            "K": calib_fields.get("K"),
            "D": D,
            "ds_params": {"fx": D[0], "fy": D[1], "cx": D[2], "cy": D[3],
                          "xi": D[4], "alpha": D[5]} if len(D) >= 6 else None,
            "R": calib_fields.get("R"),
            "P": calib_fields.get("P"),
            "T_b_c_txyz_qxyzw": calib_fields.get("T_b_c"),
            "frame_id": calib_fields.get("frame_id"),
        }
    with open(out_dir / f"calib_{camera}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if camera == "camera0":                       # 兼容旧脚本的主标定名
        with open(out_dir / "calib.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"file": str(mcap_path), "status": "ok", "frames": len(frames),
            "fps": round(fps, 3), "duration_s": round(dur, 2),
            "dropped": drop, "imu": len(imu_t), "out": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="mcap 文件或目录")
    ap.add_argument("--camera", default="camera0",
                    help="camera0 | 逗号分隔 camera0,camera1 | all（全部6目）")
    ap.add_argument("--prefix", default="/robot0", help="话题前缀")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pattern", default="DAS-Ego", help="文件名过滤（目录模式）")
    ap.add_argument("--keep-stream", action="store_true", help="保留中间 .h264")
    args = ap.parse_args()

    inp = Path(args.input)
    files = [inp] if inp.is_file() else sorted(
        p for p in inp.rglob("*.mcap") if args.pattern in p.name)
    if not files:
        print(f"no mcap matched under {inp}")
        raise SystemExit(1)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    fails = 0
    for i, p in enumerate(files):
        results = process_file(p, out_root, args.camera, args.prefix, args.keep_stream)
        line = '; '.join(f"{cam}:{r.get('status', '?')}" for cam, r in results.items())
        print(f"[{i+1}/{len(files)}] {p.name}: {line}")
        fails += sum(1 for r in results.values() if r.get('status') != 'ok')
    if fails:
        print(f"\n{fails} camera-export(s) failed")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
