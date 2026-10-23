# -*- coding: utf-8 -*-
"""DAS 多目重建统一入口：单目 / 双目 / 六目 + IMU → 双手空间轨迹。

用法（一条命令跑完全链）:
    python run_multiview.py --mode rig6 \
        --mcap '/mnt/d/work/ego/$person.1/.../DAS-Ego_xxx.mcap' \
        --start 2.4 --duration 5 --out examples/out_rig6

    python run_multiview.py --mode stereo --mcap ... --start 2.4 --duration 5 --out ...
    python run_multiview.py --mode mono   --mcap ... --start 2.4 --duration 5 --out ...

模式:
    mono    仅 camera0（原单目管线）
    stereo  camera0 + camera1（额外做立体三角化度量尺度）
    rig6    全部 6 目（有效目自动参与融合）

流程: mcap 导出(按需) → 逐目去畸变 → 15fps/456x256 预处理 + 窗口剪切
      → 逐目重建管线 → 跨目融合(配准/尺度/中值投票) → IMU 增强(重力对齐+去抖)
      → 四面板可视化视频 + Pi0.5 14 维数据集

已存在的中间产物自动跳过（按输出目录判断），可反复调用续跑。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable

MODE_CAMS = {'mono': ['camera0'], 'stereo': ['camera0', 'camera1'],
             'rig6': [f'camera{i}' for i in range(6)]}


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=list(MODE_CAMS), default='rig6')
    ap.add_argument('--mcap', required=False, default=None,
                    help='原始 mcap 路径（DAS格式）')
    ap.add_argument('--input-export', default=None,
                    help='已有导出目录(camera0.mp4,camera1.mp4,imu.npz)')
    ap.add_argument('--start', type=float, required=True, help='窗口起始秒（重采样时间线）')
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--export-root', default='/mnt/d/work/ego/das_export')
    ap.add_argument('--hfov', type=float, default=120.0)
    ap.add_argument('--task', default=None,
                    help='语义标注（任务描述），写入各 clip 任务文本并按源任务聚合')
    ap.add_argument('--vlm', action='store_true',
                    help='动作标注后用 Qwen2.5-VL 抽帧复核各动作段（需已下载模型）')
    ap.add_argument('--audit-fill', action='store_true',
                    help='跑完后审计掉帧（qc 报告）并补 ≤3 帧小洞（fused_filled.parquet）')
    ap.add_argument('--hoi', action='store_true',
                    help='生成 HOI 语义审阅视频（中文红框；含 Qwen2.5-VL 推理，约 2-3 分钟/片段）')
    ap.add_argument('--vlm-open', action='store_true',
                    help='开放动作识别：VLM 自由描述各段+全窗任务；任务框为空时自动采用其任务概括')
    ap.add_argument('--skip-pipeline', action='store_true',
                    help='各目管线已跑过时直接从融合开始')
    args = ap.parse_args()

    cams = MODE_CAMS[args.mode]
    if args.input_export:
        rec = Path(args.input_export).name
    else:
        rec = Path(args.mcap).name.rsplit('.', 1)[0]
    # 去掉末尾的会话/订阅标记（_none_none_device_hash 保留 hash）
    if args.input_export:
        export = Path(args.input_export)
    else:
        export = Path(args.export_root) / rec
    out = Path(args.out)
    rig_in = out / 'rig_input'
    t0, t1 = args.start, args.start + args.duration

    # 1) mcap 导出（mp4 与分目标定齐全才跳过，兼容旧版只导过 camera0 的目录）
    need = []

    if not args.input_export:
        need = [c for c in cams
                if not (export / f'{c}.mp4').exists()
                or not (export / f'calib_{c}.json').exists()]
    if need:
        run([PY, HERE / 'export_mcap.py', '--input', args.mcap,
             '--camera', ','.join(need), '--output', args.export_root])

    # 1b) EgoCap 等外部导出：calib 缺 K/D/T_b_c 时自动生成 fallback 双目标定
    if args.input_export:
        import json as _json
        def _calib_ok(cam):
            p = export / f'calib_{cam}.json'
            if not p.exists():
                return False
            try:
                cal = _json.loads(p.read_text(encoding='utf-8')).get('calibration') or {}
            except _json.JSONDecodeError:
                return False
            return 'K' in cal and 'T_b_c_txyz_qxyzw' in cal
        if not all(_calib_ok(c) for c in cams):
            run([PY, REPO / 'make_egocap_calibration.py',
                 '--input-export', args.input_export])

    # 2) 去畸变 + 预处理 + 窗口剪切
    rig_in.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        rect = export / f'{cam}_rect.mp4'

        # EgoCap 已经输出 pinhole 图像，无 double-sphere 畸变
        # 直接复用原视频
        if args.input_export:
            rect = export / f'{cam}.mp4'
        elif not rect.exists():
            run([PY, HERE / 'undistort_ds.py',
                 '--calib', export / f'calib_{cam}.json',
                 '--video', export / f'{cam}.mp4',
                 '--out', rect, '--hfov', args.hfov])
        cut = rig_in / f'{cam}_rig_{t0:.2f}s_{t1:.2f}s.mp4'
        if not cut.exists():
            import cv2
            cap = cv2.VideoCapture(str(rect)); w = cap.get(3); h = cap.get(4)
            cap.release()
            vf = f"fps=15,scale=456:{round(456 * h / w / 2) * 2}"
            run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', t0, '-t', args.duration,
                 '-i', rect, '-vf', vf, '-c:v', 'libx264', '-crf', '18',
                 '-preset', 'veryfast', '-pix_fmt', 'yuv420p', cut])

    # 3) 逐目重建管线
    parquets = {}
    ref = cams[0]
    if not args.skip_pipeline:
        env_export = {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
                      'EGO2LEROBOT_GPU_ALLOC_GB': '8',
                      'EGO2LEROBOT_RAY_TMP': '/tmp/ray_ego',
                      'TORCH_HOME': str(Path.home() / '.cache/torch'),
                      'TRITON_CACHE_DIR': '/tmp/triton',
                      'PYTHONUNBUFFERED': '1'}
        import os
        for cam in cams:
            cam_cut = rig_in / f'{cam}_rig_{t0:.2f}s_{t1:.2f}s.mp4'
            cam_out = out / 'views' / cam
            g = sorted(cam_out.rglob('file-000.parquet'))
            if g:
                parquets[cam] = g[0]
                continue
            run([PY, '-m', 'cli', 'pipeline',
                 '--input', cam_cut, '--output', cam_out, '--num_gpus', '1', '--test',
                 '--long_video_threshold_s', '1000000', '--per_video_lerobot',
                 '--delete_temp', 'yes'], cwd=REPO, env={**os.environ, **env_export})
            found = sorted(cam_out.rglob('file-000.parquet'))
            if found:
                parquets[cam] = found[0]
            elif cam == ref:
                raise SystemExit(f'{cam}(参考目): 管线未产出 lerobot_v3 —— 该窗口参考视角'
                                 f'无有效手部帧，请换起点/时长或录制')
            else:
                print(f'[multiview] {cam}: 本窗口无有效手部帧，跳过该视角', flush=True)
    else:
        for cam in cams:
            parquets[cam] = sorted((out / 'views' / cam).rglob('file-000.parquet'))[0]

    # 4) 融合（单目直接用参考目；被跳过的目不参与；双/六目走融合）
    cams = [c for c in cams if c in parquets]          # 剔除无产物视角
    if len(cams) < len(MODE_CAMS[args.mode]):
        print(f'[multiview] 实际参与视角: {cams}', flush=True)
    if len(cams) == 1:
        fused = parquets[ref]
    else:
        fused = out / 'fused.parquet'
        if not fused.exists():
            run([PY, HERE / 'fuse_multiview.py',
                 '--views', *[f'{c}={p}' for c, p in parquets.items()],
                 '--ref', ref, '--stereo', cams[1] if len(cams) > 1 else ref,
                 '--rig-cal', export, '--videos', rig_in, '--out', fused])

    # 5) 动作标注（规则法四类；--vlm 时追加 Qwen2.5-VL 抽帧复核）——先标注后可视化
    ref_video_cut = rig_in / f'{ref}_rig_{t0:.2f}s_{t1:.2f}s.mp4'
    act_args = ['--parquet', fused, '--video', ref_video_cut, '--export', export]
    if args.vlm:
        act_args += ['--vlm', 'qwen']
    if args.vlm_open:
        act_args += ['--vlm-open']
    run([PY, HERE / 'label_actions.py', *act_args])
    act_json = Path(fused).with_suffix('.actions.json')
    if args.vlm_open and not args.task and act_json.exists():
        import json as _json
        try:
            sug = (_json.loads(act_json.read_text(encoding='utf-8'))
                   .get('open', {}).get('task_suggest'))
        except Exception:
            sug = None
        if sug:
            args.task = sug
            print(f'[multiview] 任务描述（开放识别自动采用）: {sug}', flush=True)

    # 6) 可视化：N 输入面板(各带手部建模) + 世界系 3D（IMU 增强融合轨迹）
    #    有 .actions.json 时把动作/VLM复核/任务烧进画面顶部语义条。
    #    输出到 <out>/viz_<mode>_<时间戳>/，不覆盖历史，便于跨模式/多次对比。
    ref_video = sorted((out / 'views' / ref).rglob('file-000.mp4'))[0]
    view_args = [f'{c}={sorted((out / "views" / c).rglob("file-000.mp4"))[0]},{parquets[c]}'
                 for c in cams]
    viz_args = [PY, HERE / 'viz_multiview.py', '--tag', args.mode,
                '--views', *view_args, '--fused', fused,
                '--export', export, '--seg-start', t0,
                '--out-root', out / 'viz']
    if act_json.exists():
        viz_args += ['--actions', act_json]
    if args.task:
        viz_args += ['--task-label', args.task]
    run(viz_args)

    # 7) 语义标注 + Pi0.5 数据集（--task 提供时写入任务文本并按源任务聚合）
    if args.task:
        run([PY, HERE / 'label_tasks.py', '--run', out, '--task', args.task])
        pi05_args = ['--from-source-task']
    else:
        pi05_args = ['--prompt', 'hand manipulation on the desk']
    run([PY, REPO / 'ego_pipeline/tasks/lerobot_to_pi05.py',
         '--source', out / 'views' / ref if len(cams) == 1 else out,
         '--output', out / 'pi05', '--action-mode', 'absolute',
         *pi05_args, '--overwrite'], cwd=REPO)
    # 8) 掉帧检查 + 补帧（--audit-fill）
    fused_path = Path(fused)
    filled = fused_path.with_name(fused_path.stem + '_filled.parquet')
    if args.audit_fill and fused_path.exists():
        run([PY, HERE / 'gap_fill.py', '--parquet', fused_path])
        run([PY, REPO / 'scripts/audit_trajectory.py',
             '--parquet', str(fused_path), str(filled), '--out-dir', out / 'qc'])
        print('[OK] 掉帧审计+补帧: ' + str(out / 'qc'), flush=True)

    # 9) HOI 语义标注审阅视频（--hoi，中文红框 + 每手就近标注）
    if args.hoi and fused_path.exists():
        src = filled if (args.audit_fill and filled.exists()) else fused_path
        sd = src.parent / 'structured'
        run([PY, HERE / 'label_structured.py', '--parquet', src])
        run([PY, HERE / 'prepare_hoi_clips.py', '--structured-dir', sd,
             '--rig-input', rig_in])
        run([PY, HERE / 'label_visual.py',
             '--visual-context', sd / 'visual_context', '--export', export], cwd=HERE)
        run([PY, HERE / 'fuse_hoi.py', '--structured-dir', sd])
        run([PY, HERE / 'render_hoi_review.py', '--structured-dir', sd,
             '--lang', 'cn', '--minimal'])
        # 主视频叠加中文红框语义：重跑 viz（新时间戳目录，界面自动播放最新）
        hoi_viz_args = [PY, HERE / 'viz_multiview.py', '--tag', args.mode + '+hoi',
                        '--views', *view_args, '--fused', fused,
                        '--export', export, '--seg-start', t0,
                        '--out-root', out / 'viz',
                        '--hoi-labels', sd / 'hoi_fusion/hoi_frame_labels.parquet',
                        '--hand-boxes', sd / 'grounding/hand_boxes.parquet']
        if act_json.exists():
            hoi_viz_args += ['--actions', act_json]
        if args.task:
            hoi_viz_args += ['--task-label', args.task]
        run(hoi_viz_args)
        print(f'[OK] HOI 审阅视频(中文红框): {sd}/hoi_fusion/review_videos/', flush=True)

    print(f'\n[OK] 成品: {out}/viz/{args.mode}_<时间戳>/multiview.mp4 + {out}/pi05')


if __name__ == '__main__':
    main()
