# -*- coding: utf-8 -*-
"""单段 IMU 增强：重力对齐 + 陀螺去抖，输出对比可视化（含实时 IMU 面板）。

输入：
  --parquet   管线 lerobot_v3 的 data/chunk-000/file-000.parquet
  --video     lerobot_v3 的 observation.images.ego mp4
  --export    das_export/<该录制>/ 目录（camera0_timestamps.npz + imu.npz + calib.json）
  --seg-start 段在重采样(15fps)时间线上的起始秒（文件名 _8.20s_21.80s → 8.20）
  --out       输出目录

处理：
  1) 段帧 k(15fps) → 设备纳秒钟（经 30fps rect 视频逐帧时间戳插值）
  2) 重力对齐：比力(accel)经 R_w_c·R_c_b 映到世界系取中值 → 旋转世界使 z 轴向上
  3) 陀螺去抖：视觉角速度与陀螺角速度之差的高频分量视为抖动并剔除（端点锚定）
  4) 重建双手 EE 世界位姿（保持 T_c_ee 不变，替换相机轨迹），得到修正后 14 维 state

输出：out/imu_adjusted.npz + out/imu_enhanced.mp4 + 帧 PNG
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

G = 9.80665


def quat_xyzw_to_R(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def so3_exp(rv):
    return R.from_rotvec(np.asarray(rv).reshape(-1, 3)).as_matrix()


def compute_adjusted(parquet, export, seg_start, smooth_sigma=4.0, out=None):
    """IMU 增强计算（可被 viz_multiview 等外部调用）。

    返回 dict：state_after(14维)、res_a(derive 结果含 action/hand_valid)、
    camR_a/camt_a(校正后相机 w2c)、以及渲染/存档所需的中间量。
    """
    from types import SimpleNamespace
    args = SimpleNamespace(
        parquet=str(parquet), export=str(export),
        seg_start=float(seg_start), smooth_sigma=float(smooth_sigma),
        video='', out=str(out) if out else '/tmp/imu_enhance_out')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 读取管线产物
    df = pq.read_table(args.parquet).to_pandas()
    N = len(df)
    t_seg = df['timestamp'].to_numpy(dtype=np.float64)          # 段内秒, 15fps
    w2c = np.stack([np.asarray(v).reshape(4, 4) for v in df['extrinsics_w2c']])
    R_w_c = np.transpose(w2c[:, :3, :3], (0, 2, 1))             # R_c_w^T = R_w_c
    p_w_c = -(R_w_c @ w2c[:, :3, 3:4])[:, :, 0]                 # t_w_c = -R_w_c t_c_w

    hand = {}
    for side in ('left', 'right'):
        hand[side] = {
            'p': np.stack([np.asarray(v).reshape(3) for v in df[f'{side}_transl_world']]),
            'R': np.stack([np.asarray(v).reshape(3, 3) for v in df[f'{side}_orient_world']]),
            'kept': np.asarray(df[f'{side}_kept'].to_list(), dtype=bool)
                    if f'{side}_kept' in df else np.ones(N, bool),
        }
    state0 = np.stack([np.asarray(v).reshape(-1) for v in df['observation.state']])
    kp21 = {}
    for side in ('left', 'right'):
        cols = [f'{side}_transl_world', f'{side}_orient_world', f'{side}_kept',
                f'{side}_hand_kp21_world']
        kp = np.stack([np.asarray(v).reshape(21, 3) for v in df[f'{side}_hand_kp21_world']])
        kp21[side] = kp

    # ---------------- 读取 DAS 导出（IMU + 相机时间戳 + 外参）
    exp = Path(args.export)
    ts = np.load(exp / 'camera0_timestamps.npz')['log_time_ns'].astype(np.float64)
    imu_t_raw = np.load(exp / 'imu.npz')['t_log_ns'].astype(np.float64)
    # EgoCap 等相对时间戳：平移到 IMU 绝对钟，避免窗口匹配全部落空
    if abs(ts[0]) < 1e15 and abs(imu_t_raw[0]) > 1e15:
        ts = ts + imu_t_raw[0]
    imu = np.load(exp / 'imu.npz')
    t_imu = imu['t_log_ns'].astype(np.float64)
    w_body = imu['gyro_rad_s'].astype(np.float64)               # rad/s
    a_body = imu['accel_g'].astype(np.float64) * G               # g -> m/s^2
    _cj = exp / 'calib.json'
    if not _cj.exists() or 'T_b_c_txyz_qxyzw' not in (
            json.loads(_cj.read_text(encoding='utf-8')).get('calibration') or {}):
        _cj = exp / 'calib_camera0.json'          # EgoCap 等外部导出的回退
    calib = json.loads(_cj.read_text(encoding='utf-8'))
    tb = calib['calibration']['T_b_c_txyz_qxyzw']
    R_b_c = quat_xyzw_to_R(tb[3:7])                             # body <- camera
    R_c_b = R_b_c.T

    # 段帧 -> 设备钟：段内秒 → rect 视频时间线 → 设备 ns
    rect_t = (ts - ts[0]) / 1e9
    rect_dur = rect_t[-1]
    t_k = np.interp(args.seg_start + t_seg, rect_t, ts)          # (N,) 设备 ns
    print(f'[imu] 段 {args.seg_start:.2f}s 起, {N} 帧, 设备钟 {t_k[0]:.0f}..{t_k[-1]:.0f}')
    print(f'[imu] IMU {len(t_imu)} 采样 @ {len(t_imu)/((t_imu[-1]-t_imu[0])/1e9):.0f} Hz, '
          f'覆盖段内 IMU {(np.sum((t_imu>=t_k[0])&(t_imu<=t_k[-1])))} 条')

    dt = np.diff(t_k) / 1e9                                      # (N-1,)

    # ---------------- 0) IMU↔视频时间偏移自校准
    # 段切割/重采样让名义对齐偏差可达百余毫秒；用陀螺-视觉残差最小化校准。
    dvis0 = R.from_matrix(np.einsum('nji,njk->nik', R_w_c[:-1], R_w_c[1:])).as_rotvec()
    grid_t = np.arange(t_k[0] - 0.5e9, t_k[-1] + 0.5e9, 5e6)
    wi = np.stack([np.interp(grid_t, t_imu, w_body[:, j]) for j in range(3)], 1)
    # 陀螺累积积分：任意偏移下按帧区间积分取样（等效区间平均，抑制采样噪声）
    grid_ts = np.concatenate([np.zeros((1, 3)),
                              np.cumsum(0.5 * (wi[1:] + wi[:-1]) * 5e6 / 1e9, axis=0)])
    def gyro_inc(off_ns):
        a = np.stack([np.interp(t_k[:-1] + off_ns, grid_t, grid_ts[:, j]) for j in range(3)], 1)
        b = np.stack([np.interp(t_k[1:] + off_ns, grid_t, grid_ts[:, j]) for j in range(3)], 1)
        span = ((t_k[1:] + off_ns) - (t_k[:-1] + off_ns))[:, None]
        return (b - a) / (span / 1e9)

    best = None
    res_at_zero = None
    for off_ms in np.arange(-400, 401, 5):
        dg = gyro_inc(off_ms * 1e6) * dt[:, None]
        Dg = np.matmul(np.matmul(R_c_b, R.from_rotvec(dg).as_matrix()), R_c_b.T)
        dgyr_c = R.from_matrix(Dg).as_rotvec()
        r2 = (R.from_rotvec(dvis0) * R.from_rotvec(dgyr_c).inv()).as_rotvec()
        s = np.degrees(np.sqrt((r2 ** 2).sum(1).mean()))
        if off_ms == 0:
            res_at_zero = s
        if best is None or s < best[1]:
            best = (off_ms, s, dgyr_c)
    off_ms, res_min, dgyr = best
    t_k = t_k + off_ms * 1e6
    print(f'[imu] 时间偏移自校准: {off_ms:+.0f} ms (残差 {res_at_zero:.3f} → {res_min:.3f} °/帧)')

    # ---------------- 1) 重力对齐
    # 比力(身体系) → 世界系，均值方向 = 世界上方向
    R_wb = np.einsum('nij,jk->nik', R_w_c, R_c_b)                # (N,3,3) world<-body
    t_mid = 0.5 * (t_k[:-1] + t_k[1:])
    a_w = np.zeros((N - 1, 3))
    w_w = np.zeros((N - 1, 3))
    kIdx = np.clip(np.searchsorted(t_k, t_mid), 0, N - 2)
    for i in range(N - 1):
        m = (t_imu >= t_k[i]) & (t_imu < t_k[i + 1])
        tc = 0.5 * (t_k[i] + t_k[i + 1])
        if m.sum() == 0:                                         # 帧间无采样 → 时间插值
            ab = np.stack([np.interp(tc, t_imu, a_body[:, j]) for j in range(3)])
            wb = np.stack([np.interp(tc, t_imu, w_body[:, j]) for j in range(3)])
            a_w[i] = R_wb[kIdx[i]] @ ab
            w_w[i] = R_wb[kIdx[i]] @ wb
        else:
            a_w[i] = R_wb[kIdx[i]] @ a_body[m].mean(axis=0)
            w_w[i] = R_wb[kIdx[i]] @ w_body[m].mean(axis=0)
    up_before = a_w.mean(axis=0)
    up_before /= np.linalg.norm(up_before)
    R_align, _ = R.align_vectors([[0, 0, 1]], up_before.reshape(1, 3))
    R_align = R_align.as_matrix()
    ang_err = np.degrees(np.arccos(np.clip(up_before[2], -1, 1)))
    print(f'[imu] 对齐前世界上方向 = ({up_before[0]:+.3f},{up_before[1]:+.3f},{up_before[2]:+.3f}), '
          f'与 z 轴差 {ang_err:.2f}° → 已旋正')

    # ---------------- 2) 陀螺去抖（端点锚定）
    # 视觉帧间旋转增量（相机系小量，链式组合无大角度误差）
    dvis = R.from_matrix(np.einsum('nji,njk->nik', R_w_c[:-1], R_w_c[1:])).as_rotvec()
    # 陀螺帧间增量已在校准阶段算好（dgyr，含 R_c_b 映射）
    # 残差（视觉-陀螺），高频分量视为抖动剔除
    rvec = (R.from_rotvec(dvis) * R.from_rotvec(dgyr).inv()).as_rotvec()
    jit = rvec - gaussian_filter1d(rvec, args.smooth_sigma, axis=0)
    dfix = (R.from_rotvec(-jit) * R.from_rotvec(dvis)).as_matrix()
    R_fix = [R_w_c[0]]
    for i in range(N - 1):
        R_fix.append(R_fix[-1] @ dfix[i])
    R_fix = np.stack(R_fix)
    # 端点再锚定：把链式累积漂移沿时间线性卸载，保证首末姿态与视觉一致
    from scipy.spatial.transform import Slerp
    D = R_fix[-1] @ R_w_c[-1].T
    Dk = Slerp([0, N - 1], R.from_matrix(np.stack([np.eye(3), D])))(np.arange(N)).as_matrix()
    R_fix = np.einsum('nji,njk->nik', Dk, R_fix)
    jit_rms = np.degrees(np.sqrt((jit ** 2).sum(1).mean())) * 15
    print(f'[imu] 剔除抖动 RMS ≈ {jit_rms:.3f} °/s（视觉-陀螺高频分量）')
    # 位置轻平滑（同一时间尺度）
    p_fix = p_w_c.copy()
    for j in range(3):
        p_fix[:, j] = gaussian_filter1d(p_w_c[:, j], 1.5)

    # 全局重力旋转
    R_after = np.einsum('ij,njk->nik', R_align, R_fix)
    p_after = p_w_c @ R_align.T
    p_after = p_fix @ R_align.T

    # ---------------- 3) 重建 EE（手在相机系的位姿不变，替换相机轨迹+对齐）
    # 14 维前后对照均复用 lerobot_to_pi05 的派生：世界刚体旋转在首帧规范下自动抵消，
    # 真正改变 state 的是相机轨迹去抖；kp21 为刚体不变量（夹爪开度不受影响）。
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ego_pipeline.tasks.lerobot_to_pi05 import derive_bimanual_ee

    states_after = state0.copy()
    traj = {}
    for side in ('left', 'right'):
        pE, RE = hand[side]['p'], hand[side]['R']
        R_c_e = np.einsum('nji,njk->nik', R_w_c, RE)             # 手在相机系
        p_c_e = np.einsum('nji,nj->ni', R_w_c, pE - p_w_c)
        RE_a = np.einsum('nij,njk->nik', R_after, R_c_e)         # 新世界系
        pE_a = np.einsum('nij,nj->ni', R_after, p_c_e) + p_after
        traj[side] = (pE, RE, pE_a, RE_a)

    def _w2c(Rs, ps):
        Rc = np.transpose(Rs, (0, 2, 1))
        return Rc, np.einsum('nij,nj->ni', Rc, -ps)

    camR_b, camt_b = _w2c(R_w_c, p_w_c)
    camR_a, camt_a = _w2c(R_after, p_after)
    common = dict(left_valid=hand['left']['kept'], right_valid=hand['right']['kept'],
                  left_kp21=kp21['left'], right_kp21=kp21['right'])
    res_b = derive_bimanual_ee(
        left_xyz=traj['left'][0], right_xyz=traj['right'][0],
        left_rot=traj['left'][1], right_rot=traj['right'][1],
        cam_R=camR_b, cam_t=camt_b, **common)
    res_a = derive_bimanual_ee(
        left_xyz=traj['left'][2], right_xyz=traj['right'][2],
        left_rot=traj['left'][3], right_rot=traj['right'][3],
        cam_R=camR_a, cam_t=camt_a, **common)
    state_before = np.asarray(res_b['state'], dtype=np.float64)
    states_after = np.asarray(res_a['state'], dtype=np.float64)
    dp = np.linalg.norm(states_after[:, [0, 7]] - state_before[:, [0, 7]], axis=-1)
    def _geo_deg(rv_b, rv_a):
        Rel = R.from_rotvec(rv_b.reshape(-1, 3)).as_matrix() @ \
              R.from_rotvec(rv_a.reshape(-1, 3)).as_matrix().transpose(0, 2, 1)
        return np.degrees(np.linalg.norm(R.from_matrix(Rel).as_rotvec(), axis=-1))
    dr = np.maximum(_geo_deg(state_before[:, 3:6], states_after[:, 3:6]),
                    _geo_deg(state_before[:, 10:13], states_after[:, 10:13]))
    print(f'[imu] 14维修正幅度: 位置 Δp 中位 {np.median(dp)*1000:.1f} mm / 最大 {dp.max()*1000:.1f} mm, '
          f'姿态测地差 中位 {np.median(dr):.2f}° 最大 {dr.max():.2f}°')

    np.savez_compressed(
        out / 'imu_adjusted.npz',
        t_seg=t_seg, state_before=state_before, state_after=states_after,
        cam_p_before=p_w_c @ R_align.T, cam_p_after=p_after,
        cam_R_before=np.transpose(R_w_c, (0, 2, 1)), cam_R_after=np.transpose(R_after, (0, 2, 1)),
        R_align=R_align, up_before=up_before, jitter_deg_per_s=jit_rms,
        imu_t_ns=t_imu, imu_gyro=w_body, imu_accel=a_body,
        frame_t_ns=t_k)
    print(f'[imu] 写出 {out / "imu_adjusted.npz"}')

    return dict(out=out, df=df, N=N, t_seg=t_seg, t_k=t_k, t_mid=t_mid,
                a_w=a_w, w_w=w_w, R_align=R_align, states_after=states_after,
                state_before=state_before, res_a=res_a,
                camR_a=camR_a, camt_a=camt_a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--export', required=True)
    ap.add_argument('--seg-start', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smooth-sigma', type=float, default=4.0,
                    help='抖动判定的时间尺度(帧)，小于它的视觉-陀螺差异按抖动剔除')
    args = ap.parse_args()
    ctx = compute_adjusted(args.parquet, args.export, args.seg_start,
                           args.smooth_sigma, out=args.out)
    _render(ctx, args.video)


def _render(ctx: dict, video_path: str) -> None:
    from types import SimpleNamespace
    args = SimpleNamespace(video=video_path)
    out = ctx['out']; df = ctx['df']; N = ctx['N']; t_seg = ctx['t_seg']
    t_k = ctx['t_k']; t_mid = ctx['t_mid']; a_w = ctx['a_w']; w_w = ctx['w_w']
    R_align = ctx['R_align']; states_after = ctx['states_after']
    res_a = ctx['res_a']; camR_a = ctx['camR_a']; camt_a = ctx['camt_a']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from ego_pipeline.tasks.visualize_pi05_ee import (
        _draw_ee_overlay, _hud, _draw_3d, _pad_to_height,
        _integrate_relative, _encode_mp4, LEFT_BGR, RIGHT_BGR)

    cjk = False
    for fp in ('/mnt/c/Windows/Fonts/msyh.ttc', '/mnt/c/Windows/Fonts/simhei.ttf'):
        try:
            font_manager.fontManager.addfont(fp)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            cjk = True
            break
        except Exception:
            continue

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f'[viz] 读入 {len(frames)} 视频帧, 中文字体: {"可用" if cjk else "缺失(退回英文)"}')

    K = np.asarray(df['intrinsics'].iloc[0], dtype=np.float64).reshape(3, 3)
    cam_adj = {'K': K, 'R': camR_a, 't': camt_a}
    state_a = states_after.astype(np.float32)
    action_a = np.asarray(res_a['action'], dtype=np.float32)
    valid_a = res_a['hand_valid']
    recon_a = _integrate_relative(state_a, action_a)

    a_w_al = a_w @ R_align.T
    w_w_al = w_w @ R_align.T
    t_imu_seg = (t_mid - t_k[0]) / 1e9

    PANEL_H = 640
    LEFT_W = int(round(frames[0].shape[1] * PANEL_H / frames[0].shape[0]))
    RIGHT_W = LEFT_W
    BOT_H = 300
    CURVE_COLORS = ((25, 118, 210), (46, 160, 67), (224, 49, 49))   # BGR

    def imu_panel(k, data, deg, title, ref_line=None, note=''):
        """下排 IMU 面板: 三轴曲线 + 当前帧游标, 返回 BGR 图."""
        fig = plt.figure(figsize=(LEFT_W / 100, BOT_H / 100), dpi=100)
        ax = fig.add_subplot(111)
        plot = np.degrees(data) if deg else data
        for j, c in enumerate(CURVE_COLORS):
            ax.plot(t_imu_seg, plot[:, j], color=np.array(c)[::-1] / 255.0,
                    lw=1.0, label='xyz'[j])
        if ref_line is not None:
            ax.axhline(ref_line, color='0.5', lw=0.8, ls=':')
        ax.axvline(t_seg[k], color='k', lw=1.2, ls='--')
        cur = np.stack([np.interp(t_seg[k], t_imu_seg, plot[:, j]) for j in range(3)])
        cur_str = '  '.join(f"{'xyz'[j]}={cur[j]:+6.1f}" for j in range(3))
        ax.set_title(f'{title}   当前 {cur_str}{note}', fontsize=10, loc='left')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        ax.set_xlabel('t (s)', fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
        plt.close(fig)
        return buf

    out_frames = []
    for k in range(N):
        vis = frames[k].copy()
        _draw_ee_overlay(vis, state_a[k, :3], state_a[k, 3:6], state_a[k, 6],
                         bool(valid_a[k, 0]), LEFT_BGR, K, cam_adj, k,
                         action_a[k, :3], tag='L')
        _draw_ee_overlay(vis, state_a[k, 7:10], state_a[k, 10:13], state_a[k, 13],
                         bool(valid_a[k, 1]), RIGHT_BGR, K, cam_adj, k,
                         action_a[k, 7:10], tag='R')
        _hud(vis, k, N, state_a[k], action_a[k], valid_a[k])
        left = _pad_to_height(vis, PANEL_H)
        right = _draw_3d(PANEL_H, RIGHT_W, state_a, recon_a, action_a, valid_a, k)
        gyro_p = imu_panel(k, w_w_al, True,
                           '陀螺仪 ω (世界系, °/s) — IMU 时间基已自校准')
        acc_p = imu_panel(k, a_w_al, False,
                          '加速度 a (世界系, m/s²)', ref_line=-G,
                          note='   (az≈-9.8 → 重力已对齐)')
        top = np.concatenate([left, right], axis=1)
        bot = np.concatenate([gyro_p, acc_p], axis=1)
        row = np.concatenate([top, bot], axis=0)
        h, w = row.shape[:2]
        if h % 2 or w % 2:
            row = cv2.copyMakeBorder(row, 0, h % 2, 0, w % 2,
                                     cv2.BORDER_CONSTANT, value=(12, 12, 12))
        out_frames.append(row)
        if k % 40 == 0:
            print(f'[viz] {k}/{N}')

    mp4 = out / 'imu_enhanced.mp4'
    _encode_mp4(out_frames, mp4, 15.0)
    (out / 'snapshots').mkdir(exist_ok=True)
    for t in sorted({0, N // 3, 2 * N // 3, N - 1}):
        cv2.imwrite(str(out / 'snapshots' / f'frame_{t:04d}.png'), out_frames[t])
    print(f'[viz] 写出 {mp4}')


if __name__ == '__main__':
    main()
