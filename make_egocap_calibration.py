# -*- coding: utf-8 -*-
"""生成 EgoCap → DAS stereo 管线兼容的标定文件。

EgoCap 的 /egocap/camera/{left,right}/camera_info 里 K 全 0、D 为空，
不可用；本脚本用 fallback 双目标定覆盖写 calib_camera{0,1}.json：
  - camera0 = 立体参考目（T_b_c = 单位）
  - camera1 = 右移 6cm 基线（T_b_c 平移 [-0.06,0,0]）
同时确保 camera{N}_rect.mp4 存在（EgoCap 视频已是无畸变针孔，直接拷贝）
并写配套 camera{N}_rect.mp4.json（fuse_multiview 读取 K 用）。

用法:
    python make_egocap_calibration.py [--input-export examples/egocap_das_export/bag_0]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

WIDTH, HEIGHT = 640, 400
K = [[576.0, 0.0, 320.0],
     [0.0, 576.0, 200.0],
     [0.0, 0.0, 1.0]]
D = [0.0, 0.0, 0.0, 0.0, 0.0]
T_REF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]            # tx,ty,tz, qx,qy,qz,qw
T_BASELINE = [-0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

EXTRINSICS = {'camera0': T_REF, 'camera1': T_BASELINE}


def make(input_export: Path) -> list[Path]:
    written = []
    for cam, T in EXTRINSICS.items():
        calib_path = input_export / f'calib_{cam}.json'
        meta = {}
        if calib_path.exists():                       # 保留 EgoCap 元数据字段
            try:
                meta = json.loads(calib_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                meta = {}
        cal = dict(meta.get('calibration') or {})
        cal.update({'width': WIDTH, 'height': HEIGHT,
                    'K': K, 'D': D, 'T_b_c_txyz_qxyzw': T})
        meta.update({'source': meta.get('source', 'EgoCap'),
                     'camera': cam, 'calibration': cal})
        calib_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                              encoding='utf-8')
        written.append(calib_path)

        # rect 视频：不存在则直通拷贝（EgoCap 已是针孔），并写 fuse 用的 K json
        raw = input_export / f'{cam}.mp4'
        rect = input_export / f'{cam}_rect.mp4'
        if raw.exists() and not rect.exists():
            shutil.copy2(raw, rect)
        rect_json = input_export / f'{cam}_rect.mp4.json'
        rj = {'model': 'pinhole', 'width': WIDTH, 'height': HEIGHT, 'K': K,
              'hfov_deg': None,
              'note': 'EgoCap passthrough (already rectilinear); fallback K'}
        rect_json.write_text(json.dumps(rj, indent=2), encoding='utf-8')
        written.append(rect_json)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-export', default='examples/egocap_das_export/bag_0')
    args = ap.parse_args()
    for p in make(Path(args.input_export)):
        print(f'[egocalib] 写出 {p}')


if __name__ == '__main__':
    main()
