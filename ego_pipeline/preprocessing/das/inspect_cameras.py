# -*- coding: utf-8 -*-
"""DAS-Ego mcap 相机选型：解码全部 camera_info，用 T_b_c 外参评估朝向。

用法:
    python inspect_cameras.py /path/to/DAS-Ego_xxx.mcap

输出每路相机的内参/畸变参数/光轴方向（机体系），按"前向程度"排序。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def mat3(vals, name):
    a = np.asarray(vals, dtype=np.float64)
    if a.size != 9:
        raise ValueError(f"{name}: expect 9 elems, got {a.size}")
    return a.reshape(3, 3)


def tvec(vals, name):
    a = np.asarray(vals, dtype=np.float64)
    if a.size != 3:
        raise ValueError(f"{name}: expect 3 elems, got {a.size}")
    return a


def main(path: str) -> None:
    rows = []
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if not topic.endswith("/camera_info"):
                continue
            cam = topic.split("/")[-2] if topic.count("/") >= 2 else topic
            msg = message.protobuf  # DecoderFactory 已解码
            fields = {fd.name: v for fd, v in msg.ListFields()}

            # 畸变参数：兼容 D1..D8 标量与 repeated D 两种布局
            dvals = []
            for i in range(1, 9):
                v = fields.get(f"D{i}")
                if v is not None:
                    dvals.append(float(v))
            if "D" in fields and not dvals:
                dvals = [float(x) for x in fields["D"]]

            K = mat3(fields.get("K", np.zeros(9)), f"{cam} K")
            T_b_c = np.eye(4)
            T_b_c[:3, :3] = mat3(fields.get("T_b_c", np.eye(9)), f"{cam} T_b_c.R")
            T_b_c[:3, 3] = tvec(fields.get("T_b_c_t", fields.get("T_b_c.t", [0, 0, 0])), f"{cam} T_b_c.t") \
                if ("T_b_c_t" in fields or "T_b_c.t" in fields) else np.zeros(3)

            rows.append({
                "cam": cam, "topic": topic,
                "model": fields.get("distortion_model", "?"),
                "width": fields.get("width"), "height": fields.get("height"),
                "K": K, "D": dvals,
                "T_b_c": T_b_c,
            })

    if not rows:
        print("no camera_info found")
        return

    # 机体系约定未知，先把各相机光轴（相机系 +Z）投到机体系，看相对分布
    print(f"{'camera':<10} {'model':<8} {'size':<12} "
          f"{'axis@body(x,y,z)':<28} {'focal(fx,fy)':<20} D")
    ranked = []
    for r in rows:
        axis = r["T_b_c"][:3, :3] @ np.array([0.0, 0.0, 1.0])
        ranked.append((axis, r))
    for axis, r in ranked:
        fx, fy = r["K"][0, 0], r["K"][1, 1]
        print(f"{r['cam']:<10} {str(r['model']):<8} {r['width']}x{r['height']:<7} "
              f"({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f})   "
              f"({fx:.1f},{fy:.1f})        "
              f"{[round(d, 4) for d in r['D']]}")

    # T_b_c 平移部分（相机中心在机体系的位置）
    print("\nT_b_c translations:")
    for _, r in rows:
        t = r["T_b_c"][:3, 3]
        print(f"  {r['cam']:<10} t=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})")

    print("\nrotation matrices (body<-camera):")
    for _, r in rows:
        print(f"  {r['cam']:<10}")
        for line in r["T_b_c"][:3, :3]:
            print("    ", " ".join(f"{v:+.4f}" for v in line))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
