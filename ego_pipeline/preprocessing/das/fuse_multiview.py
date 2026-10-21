# -*- coding: utf-8 -*-
"""多目融合：把各目单目管线的结果配准到参考目规范，融合成一条双手轨迹。

输入（每目一份 lerobot_v3 data parquet + 本目 rect 标定）：
  --views   cameraN=parquet路径 的多次传入，如 camera0=.../file-000.parquet
  --ref     参考目（默认 camera0），融合结果落在它的首帧规范
  --rig-cal  各目标定目录（das_export/<rec>，含 calib_cameraN.json 与 *_rect.mp4.json）
  --videos  各目窗口视频目录（用于立体检测；文件名 cameraN_*.mp4）
  --out     融合 parquet 输出路径

流程：
  1) 度量尺度：参考目与 --stereo 目(默认 camera1)按刚体 T_b_c 求相对位姿，
     YOLO 检测双腕中心 → cv2 三角化 → 与单目深度的中值比 = 尺度因子
  2) 跨目配准：非参考目在共同有效帧上用 Umeyama(带尺度) 腕点把它的世界系
     对齐到参考目世界系（刚体 rig 下等价于用 T_b_c+IMU，且对漂移更稳）
  3) 融合：逐帧逐手中值投票（位置），姿态取参考目优先、否则四元数平均；
     kept = 任一目有效；记录 n_views
  4) 全局尺度应用到所有世界系平移（含相机中心与 kp21），输出融合 parquet
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

WRIST_KEYS = ('left_transl_world', 'right_transl_world')


def quat_xyzw_to_R(q):
    q = np.asarray(q, float); q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src, dst):
    """相似变换 dst ≈ s·R@src + t，返回 (R, s, t)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, _, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d])
    Rm = U @ E @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = np.trace(np.diag([1, 1, d]) @ np.diag(np.linalg.svd(cov)[1])) / var_s
    return Rm, s, mu_d - s * Rm @ mu_s


def load_view(parquet_path):
    df = pq.read_table(parquet_path).to_pandas()
    v = {'df': df, 'n': len(df)}
    for side in ('left', 'right'):
        v[f'{side}_p'] = np.stack([np.asarray(x).reshape(3)
                                   for x in df[f'{side}_transl_world']])
        v[f'{side}_R'] = np.stack([np.asarray(x).reshape(3, 3)
                                   for x in df[f'{side}_orient_world']])
        v[f'{side}_ok'] = np.asarray(df[f'{side}_kept'].to_list(), bool)
    v['w2c'] = np.stack([np.asarray(x).reshape(4, 4) for x in df['extrinsics_w2c']])
    v['kp'] = {s: np.stack([np.asarray(x).reshape(21, 3)
                            for x in df[f'{s}_hand_kp21_world']]) for s in ('left', 'right')}
    return v


def wrist_uv(video, model):
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = model.predict(f, verbose=False, conf=0.3, imgsz=640)[0]
        ul = ur = None
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy()
            m0, m1 = cls == 0, cls == 1
            if m0.any():
                b = xy[m0][0]; ul = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if m1.any():
                b = xy[m1][0]; ur = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        out.append((ul, ur))
    cap.release()
    return out


def align_len(v, n):
    """把非参考目对齐到参考目帧数（各目 ffmpeg 重采样偶有 ±1 帧差）。

    多出的帧截断；不足的帧补 NaN 且 ok=False（不参与配准与融合）。"""
    m = v['n']
    if m == n:
        return v
    print(f'[fuse] 帧数不对齐: {m} vs 参考 {n}，已截断/补无效帧')
    for key in ('left_p', 'right_p', 'left_R', 'right_R', 'w2c'):
        a = v[key]
        if m > n:
            v[key] = a[:n]
        else:
            v[key] = np.concatenate(
                [a, np.full((n - m, *a.shape[1:]), np.nan, dtype=a.dtype)], axis=0)
    for key in ('left_ok', 'right_ok'):
        a = v[key]
        v[key] = a[:n] if m > n else np.concatenate([a, np.zeros(n - m, bool)], axis=0)
    for s, a in list(v['kp'].items()):
        v['kp'][s] = a[:n] if m > n else np.concatenate(
            [a, np.full((n - m, 21, 3), np.nan, dtype=a.dtype)], axis=0)
    if m > n:
        v['df'] = v['df'].iloc[:n].reset_index(drop=True)
    v['n'] = n
    return v


def stereo_scale(ref_view, stereo_cal_dir, video_dir, ref_cam, stereo_cam,
                 conf_thr=0.0):
    """三角化双腕求参考目单目世界 → 米制 的尺度（返回 descale=1/s0）。"""
    from ultralytics import YOLO
    repo_root = Path(__file__).resolve().parents[3]
    model = YOLO(str(repo_root / 'model' / 'hawor' / 'detector.pt'))
    Ks = {}
    for cam in (ref_cam, stereo_cam):
        rect_json = glob.glob(f'{stereo_cal_dir}/{cam}_rect.mp4.json')

        if rect_json:
            j = json.loads(Path(rect_json[0]).read_text())
        else:
            # EgoCap pinhole camera，无 DS 畸变文件
            calib_json = glob.glob(f'{stereo_cal_dir}/calib_{cam}.json')

            if calib_json:
                j = json.loads(Path(calib_json[0]).read_text())
            else:
                raise FileNotFoundError(
                    f'No calibration found for {cam}'
                )
        if 'K' in j:
            K = np.array(j['K'], float)
        elif 'calibration' in j and 'K' in j['calibration']:
            K = np.array(j['calibration']['K'], float)
        else:
            # EgoCap fallback:
            # 640x400 stereo camera approximate intrinsics
            w = j.get('width', 640)
            h = j.get('height', 400)

            fx = fy = 0.9 * max(w, h)
            cx = w / 2.0
            cy = h / 2.0

            K = np.array([
                [fx,0,cx],
                [0,fy,cy],
                [0,0,1]
            ], dtype=float)

            print(
                '[fuse] warning: missing K, use approximate intrinsics',
                K
            )
        v = glob.glob(f'{video_dir}/{cam}_*.mp4')[0]
        cap = cv2.VideoCapture(v); w = cap.get(3); h = cap.get(4); cap.release()
        sx, sy = w / j['width'], h / j['height']
        K[0] *= sx
        K[1] *= sy
        Ks[cam] = K
    cals = {}

    for c in (ref_cam, stereo_cam):

        candidates = [
            f'{stereo_cal_dir}/calib_{c}.json',
            f'{stereo_cal_dir}/{c}_rect.mp4.json'
        ]

        cal_file = None
        for x in candidates:
            if Path(x).exists():
                cal_file = x
                break

        if cal_file is None:
            raise FileNotFoundError(
                f'cannot find calibration for {c}'
            )

        data = json.loads(Path(cal_file).read_text())

        # DAS format
        if 'calibration' in data:
            data = data['calibration']

        # EgoCap format
        if 'T_b_c_txyz_qxyzw' not in data:
            data['T_b_c_txyz_qxyzw'] = [
                0,0,0,
                0,0,0,1
            ]

        cals[c] = data

        print(
            '[fuse calib]',
            c,
            cal_file,
            data.keys()
        )
    Tb = {}
    for c, cal in cals.items():
        t, q = cal['T_b_c_txyz_qxyzw'][:3], cal['T_b_c_txyz_qxyzw'][3:]
        T = np.eye(4); Tb[c] = T
        T[:3, :3] = quat_xyzw_to_R(q); T[:3, 3] = t
    Trel = np.linalg.inv(Tb[stereo_cam]) @ Tb[ref_cam]      # stereo_cam <- ref_cam
    P0 = Ks[ref_cam] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = Ks[stereo_cam] @ Trel[:3]

    uv0 = wrist_uv(glob.glob(f'{video_dir}/{ref_cam}_*.mp4')[0], model)
    uv1 = wrist_uv(glob.glob(f'{video_dir}/{stereo_cam}_*.mp4')[0], model)
    ratios = []
    for t, ((u0l, u0r), (u1l, u1r)) in enumerate(zip(uv0, uv1)):
        if t >= ref_view['n']:
            break
        for (u0, u1), side in (((u0l, u1l), 'left'), ((u0r, u1r), 'right')):
            if u0 is None or u1 is None or not ref_view[f'{side}_ok'][t]:
                continue
            p0 = np.array([[u0[0]], [u0[1]]], np.float32)
            p1 = np.array([[u1[0]], [u1[1]]], np.float32)
            X = cv2.triangulatePoints(P0, P1, p0, p1)
            X = X[:3] / X[3]
            if not (0.05 < X[2] < 5.0):
                continue
            z_mono = (ref_view['w2c'][t, :3, :3] @ ref_view[f'{side}_p'][t]
                      + ref_view['w2c'][t, :3, 3])[2]
            if z_mono > 1e-6:
                ratios.append(z_mono / X[2])
    s0 = float(np.median(ratios)) if ratios else 1.0
    return s0, len(ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', nargs='+', required=True,
                    help='cameraN=parquet 路径（参考目须在内）')
    ap.add_argument('--ref', default='camera0')
    ap.add_argument('--stereo', default='camera1', help='用于度量尺度的第二目')
    ap.add_argument('--rig-cal', required=True, help='das_export/<rec> 目录')
    ap.add_argument('--videos', required=True, help='各目窗口视频目录')
    ap.add_argument('--out', required=True, help='融合 parquet 输出')
    args = ap.parse_args()

    views = {}
    for spec in args.views:
        cam, path = spec.split('=', 1)
        views[cam] = load_view(path)
    ref = views[args.ref]
    n = ref['n']
    for cam, v in views.items():
        if cam != args.ref:
            align_len(v, n)
    print(f'[fuse] {len(views)} 目: {sorted(views)}，参考={args.ref}，{n} 帧')

    # 1) 度量尺度
    s0, n_used = stereo_scale(ref, args.rig_cal, args.videos, args.ref, args.stereo)
    if n_used < 3:
        print(f'[fuse] 警告: 三角化样本仅 {n_used} 个（<3），尺度不可靠，回退 1.0')
        s0 = 1.0
    print(f'[fuse] 立体尺度: 单目世界 = {s0:.3f} × 米制（{n_used} 个三角化样本）')

    # 2) 跨目配准 + 3) 融合
    aligned = {args.ref: ref}
    for cam, v in views.items():
        if cam == args.ref:
            continue
        m = (v['left_ok'] & ref['left_ok']) | (v['right_ok'] & ref['right_ok'])
        pairs = []
        for side in ('left', 'right'):
            mm = m & v[f'{side}_ok'] & ref[f'{side}_ok']
            pairs.append((v[f'{side}_p'][mm], ref[f'{side}_p'][mm]))
        src = np.concatenate([p[0] for p in pairs])
        dst = np.concatenate([p[1] for p in pairs])
        if len(src) < 10:
            print(f'[fuse] {cam}: 共同有效帧不足({len(src)})，弃用')
            continue
        Rm, s, t = umeyama(src, dst)
        v['left_p'] = s * (v['left_p'] @ Rm.T) + t
        v['right_p'] = s * (v['right_p'] @ Rm.T) + t
        v['left_R'] = np.einsum('ij,njk->nik', Rm, v['left_R'])
        v['right_R'] = np.einsum('ij,njk->nik', Rm, v['right_R'])
        for side in ('left', 'right'):
            v['kp'][side] = s * (v['kp'][side] @ Rm.T) + t
        aligned[cam] = v
        resid = np.sqrt(((src @ Rm.T * s + t - dst) ** 2).sum(1)).mean()
        print(f'[fuse] {cam}: 配准完成 s={s:.3f} 腕点残差 {resid*1000:.1f} mm')

    df = ref['df'].copy()
    n_views = np.zeros(n, int)
    out = {}
    for side in ('left', 'right'):
        P = np.full((n, 3), np.nan)
        Q = np.full((n, 4), np.nan)
        kept = np.zeros(n, bool)
        for t in range(n):
            ps, qs, ok = [], [], False
            for cam, v in aligned.items():
                if v[f'{side}_ok'][t]:
                    ps.append(v[f'{side}_p'][t])
                    qs.append(R.from_matrix(v[f'{side}_R'][t]).as_quat())
                    ok = True
            if ok:
                P[t] = np.median(np.stack(ps), 0)
                kept[t] = True
                n_views[t] = len(ps)
                if len(ps) == 1:
                    Q[t] = qs[0]
                else:                        # 四元数平均（处理符号翻转）
                    Qs = np.stack(qs)
                    Qs = Qs * np.sign(Qs @ Qs[0])[:, None]
                    q = Qs.mean(0); Q[t] = q / np.linalg.norm(q)
        Q[np.isnan(Q).any(axis=1)] = [0.0, 0.0, 0.0, 1.0]   # 无效帧填单位四元数
        out[side] = (P, R.from_quat(Q).as_matrix(), kept)

    s_inv = 1.0 / s0
    df['left_kept'] = list(out['left'][2])
    df['right_kept'] = list(out['right'][2])
    def _rows(arr):
        return [np.asarray(x).reshape(-1).tolist() for x in arr]

    for side in ('left', 'right'):
        P, Rm, _ = out[side]
        df[f'{side}_transl_world'] = _rows(P * s_inv)
        df[f'{side}_orient_world'] = [x.reshape(-1).tolist() for x in Rm]
        kp = ref['kp'][side] * s_inv
        df[f'{side}_hand_kp21_world'] = [x.reshape(-1).tolist() for x in kp]
    w2c = ref['w2c'].copy()
    w2c[:, :3, 3] *= s_inv
    df['extrinsics_w2c'] = _rows(w2c)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    ref_tbl = pq.read_table(args.views[0].split('=', 1)[1])
    tbl = pa.Table.from_pandas(df, schema=ref_tbl.schema)
    pq.write_table(tbl, args.out)
    print(f'[fuse] 写出 {args.out}  '
          f'左有效 {int(out["left"][2].sum())}/{n} 右有效 {int(out["right"][2].sum())}/{n}  '
          f'平均参与目数 {n_views[n_views > 0].mean():.1f}')


if __name__ == '__main__':
    main()
