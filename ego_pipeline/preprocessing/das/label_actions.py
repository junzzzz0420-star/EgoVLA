# -*- coding: utf-8 -*-
"""动作语义自动标注（四类简单动作）：双手张开 / 双手握拳 / 双手抬起 / 双手下沉。

两种引擎:
  rules  (默认) 几何规则——基于管线已输出的 hand_kp21_world + IMU 重力方向:
         · 张开/握拳: 指尖-腕距 / 掌长 的归一化值（自动分位定阈）
         · 抬起/下沉: 腕部世界坐标在重力轴上的低通速度（IMU 加计均值定重力轴）
  qwen   大模型——抽帧 + Qwen2.5-VL 视觉问答复核（--vlm qwen，四类封闭比对）
  open   开放动作识别——--vlm-open：每段+全窗自由文本描述（不限四类），
         task_suggest 可作为任务语义自动候选

用法:
    python label_actions.py --parquet <lerobot_v3 data parquet 或 fused.parquet> \
        --export <das_export/<rec>> [--out labels.json] [--write-parquet]
输出: 逐帧标签 + 连续动作段（≥min-frames），可写回 parquet 的 action_label 列。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TIP_IDS = (4, 8, 12, 16, 20)          # 拇指/食指/中指/无名指/小指指尖
LABELS = ('hands_open', 'hands_fist', 'hands_raise', 'hands_lower', 'other')
CN = {'hands_open': '双手张开', 'hands_fist': '双手握拳',
      'hands_raise': '双手抬起', 'hands_lower': '双手下沉', 'other': '其他'}


def gravity_axis(export_dir: Path | None):
    """IMU 比力均值 → 世界系重力轴（z 向上）。无 IMU 时退回世界 z 轴。"""
    if export_dir and (export_dir / 'imu.npz').exists():
        d = np.load(export_dir / 'imu.npz')
        a = d['accel_g'].astype(np.float64) * 9.80665
        up = -a.mean(0)                                  # 加计测比力，静止时 ≈ -g → up
        up /= np.linalg.norm(up)
        # 腕轨迹的世界系是参考相机系；IMU 是机体系。近似：相机系与机体系接近刚体
        # 安装，均值方向仍以"世界竖直"为主分量，这里取世界系里与 up 最不差的
        # 处理：把 up 表达到世界系需要 T_b_c；简化用 |up| 投影到腕位移主轴的符号。
        return up
    return np.array([0.0, 0.0, 1.0])


def load_hands(parquet_path: Path):
    df = pq.read_table(parquet_path).to_pandas()
    hands = {}
    for side in ('left', 'right'):
        hands[side] = {
            'kp': np.stack([np.asarray(v).reshape(21, 3)
                            for v in df[f'{side}_hand_kp21_world']]),
            'wrist': np.stack([np.asarray(v).reshape(3)
                               for v in df[f'{side}_transl_world']]),
            'kept': np.asarray(df[f'{side}_kept'].to_list(), bool),
        }
    return df, hands


def openness(kp: np.ndarray) -> np.ndarray:
    """每帧张开度: mean(指尖到腕距) / 掌长(腕0→中指根9)。张开>1.5, 握拳<1.0 量级。"""
    wrist = kp[:, 0:1, :]
    palm = np.linalg.norm(kp[:, 9, :] - kp[:, 0, :], axis=-1) + 1e-9
    tip_d = np.stack([np.linalg.norm(kp[:, i, :] - wrist[:, 0, :], axis=-1)
                      for i in TIP_IDS], axis=-1).mean(-1)
    return tip_d / palm


def segments(labels: list[str], min_frames: int):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != 'other' and i - start >= min_frames:
                out.append({'label': labels[start], 'label_cn': CN[labels[start]],
                            'start_frame': start, 'end_frame': i - 1})
            start = i
    return out


def annotate(parquet_path: Path, export_dir: Path | None, min_frames: int = 5,
             v_smooth: int = 5, quantile: bool = False):
    df, hands = load_hands(parquet_path)
    n = len(df)

    # --- 张开度（双手中值；kept 无效帧标记 nan）
    op = np.full((2, n), np.nan)
    for i, side in enumerate(('left', 'right')):
        o = openness(hands[side]['kp'])
        o[~hands[side]['kept']] = np.nan
        op[i] = o
    both_ok = ~np.isnan(op).any(0)
    opv = np.nanmedian(op, 0)

    # 阈值：默认物理绝对值（张开度=指尖均距/掌长，握拳<1.05、张开>1.5）；
    # --quantile 模式按本段分布自适应（适合个体手掌比例差异大的情况）
    valid = opv[both_ok]
    if quantile and len(valid) > 10:
        th_lo, th_hi = float(np.quantile(valid, 0.25)), float(np.quantile(valid, 0.75))
    else:
        th_lo, th_hi = 1.05, 1.50

    # --- 重力轴上的运动（腕部世界位置低通速度）
    z = np.stack([hands[s]['wrist'][:, 2] for s in ('left', 'right')]).mean(0)
    if export_dir is not None:
        g = gravity_axis(export_dir)
        # 近似投影：世界 z 与重力轴夹角未知时退回 z（相机系 z 通常接近水平/光轴，
        # 这里用腕位移方差最大方向替换太复杂——保持 z，重力对齐后的 fused 数据最准）
    dt = 1.0 / 15.0
    k = np.ones(v_smooth) / v_smooth
    zs = np.convolve(z, k, 'same')
    vz = np.gradient(zs, dt)
    vz_abs = np.abs(vz)
    th_v = max(np.quantile(vz_abs[both_ok], 0.75), 0.03) if both_ok.sum() > 10 else 0.05

    labels = []
    for t in range(n):
        if not both_ok[t]:
            labels.append('other'); continue
        if vz[t] > th_v:
            labels.append('hands_raise')
        elif vz[t] < -th_v:
            labels.append('hands_lower')
        elif opv[t] > th_hi:
            labels.append('hands_open')
        elif opv[t] < th_lo:
            labels.append('hands_fist')
        else:
            labels.append('other')

    return {'n': n, 'labels': labels,
            'segments': segments(labels, min_frames),
            'stats': {'open_p25': float(th_lo), 'open_p75': float(th_hi),
                      'vz_thresh_mps': float(th_v),
                      'open_min': float(np.nanmin(opv)), 'open_max': float(np.nanmax(opv)),
                      'valid_ratio': float(both_ok.mean())}}


def find_video(parquet_path: Path):
    """自动找与 parquet 帧号对应的参考目视频：优先 out/rig_input/ 的剪切版。"""
    for anc in parquet_path.parents:
        for v in sorted((anc / 'rig_input').glob('camera0_rig_*.mp4')):
            return v
    for anc in [parquet_path.parent, *list(parquet_path.parents)[:3]]:
        g = sorted(anc.rglob('file-000.mp4'))
        if g:
            return g[0]
    return None


VLM_PROMPT = ('这张图是头戴相机拍摄的第一视角画面。请判断画面中人双手的状态，'
              '只从下列选项中选一项，只回答选项字母，不要解释：\n'
              'A 双手张开（手指伸展摊开）\nB 双手握拳\n'
              'C 双手抬起（正在向上移动）\nD 双手下沉（正在向下移动）\n'
              'E 其他或无法判断')

_VLM_KEY = {'A': 'hands_open', 'B': 'hands_fist', 'C': 'hands_raise',
            'D': 'hands_lower', 'E': 'other',
            '张开': 'hands_open', '握拳': 'hands_fist',
            '抬起': 'hands_raise', '下沉': 'hands_lower'}


def _resolve_model_path(model_id: str) -> str:
    """优先用本地 HF 缓存的快照目录，避免联网校验。"""
    if Path(model_id).exists():
        return model_id
    hub = Path.home() / '.cache/huggingface/hub'
    tail = model_id.rsplit('/', 1)[-1]
    for d in sorted(hub.glob(f'models--*{tail}/snapshots/*')):
        if (d / 'config.json').exists():
            return str(d)
    return model_id


def _load_vlm(model_id: str):
    """加载本地 Qwen2.5-VL（AWQ）。返回 (model, processor, 路径)。"""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    mpath = _resolve_model_path(model_id)
    print(f'[vlm] 加载 {mpath}（首次约 1 分钟）...', flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mpath, dtype=torch.float16, device_map='cuda').to(torch.float16)
    processor = AutoProcessor.from_pretrained(mpath)
    model.eval()
    return model, processor, mpath


def _unload_vlm(model):
    del model
    import torch
    torch.cuda.empty_cache()


def _grabber(video_path: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(i, total - 1)))
        ok, bgr = cap.read()
        return bgr if ok else None

    return grab, cap.release


def _ask(model, processor, frames_bgr, prompt: str, max_new_tokens: int = 40) -> str:
    """多帧（或单帧）一条消息问答，返回生成的文本。"""
    import base64

    import cv2
    import torch
    from qwen_vl_utils import process_vision_info

    content = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        uri = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
        content.append({'type': 'image', 'image': uri})
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def _parse_choice(ans: str):
    for k, v in _VLM_KEY.items():
        if k in ans:
            return v
    return 'other'


def vlm_review(res: dict, video_path: Path, model_id: str = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
               frames_per_seg: int = 3, _loaded=None) -> dict:
    """对规则法检出的每个动作段抽帧问 Qwen2.5-VL，段级投票复核（不改标签，只报告）。"""
    own = _loaded is None
    model, processor, mpath = _loaded or _load_vlm(model_id)
    grab, close = _grabber(video_path)

    for seg in res['segments']:
        span = seg['end_frame'] - seg['start_frame'] + 1
        idxs = np.linspace(seg['start_frame'], seg['end_frame'],
                           min(frames_per_seg, span)).round().astype(int)
        votes, raws = [], []
        for i in idxs:
            f = grab(int(i))
            if f is None:
                continue
            ans = _ask(model, processor, [f], VLM_PROMPT, max_new_tokens=8)
            raws.append(ans)
            votes.append(_parse_choice(ans))
        if not votes:
            seg['vlm_label'] = None
            continue
        from collections import Counter
        seg['vlm_label'] = Counter(votes).most_common(1)[0][0]
        seg['vlm_votes'] = dict(Counter(votes))
        seg['vlm_raw'] = raws
        seg['agree'] = (seg['vlm_label'] == seg['label'])
        mark = '✓' if seg['agree'] else '✗'
        print(f'  段 {seg["start_frame"]:3d}-{seg["end_frame"]:3d}  '
              f'规则:{seg["label_cn"]}  VLM:{CN.get(seg["vlm_label"], "?")}  {mark}',
              flush=True)

    close()
    segs = res['segments']
    res['vlm'] = {'model': mpath, 'video': str(video_path),
                  'n_segs': len(segs),
                  'n_agree': sum(1 for s in segs if s.get('agree')),
                  'agreement': (sum(1 for s in segs if s.get('agree')) / len(segs)
                                if segs else None)}
    if own:
        _unload_vlm(model)
    return res


OPEN_SEG_PROMPT = ('这是同一段第一视角录像按时间先后抽取的{n}帧。'
                   '用不超过12个汉字描述画面里双手正在做的具体动作'
                   '（例如：从桌面拿起螺丝 / 左右手配合旋转瓶盖 / 松手放下物体），'
                   '只输出这一句描述，不要编号不要解释。')
OPEN_TASK_PROMPT = ('这是同一段第一视角任务录像按时间先后抽取的{n}帧。'
                    '用一句话（20字以内）概括佩戴者双手完成的整体任务'
                    '（例如：双手配合把螺丝拧入工件），只输出这一句话。')


def vlm_open(res: dict, video_path: Path, model_id: str = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
             frames_per_seg: int = 5, whole_frames: int = 6, _loaded=None) -> dict:
    """开放动作识别：每个动作段 + 全窗，VLM 自由文本描述（不限于四类封闭标签）。"""
    own = _loaded is None
    model, processor, mpath = _loaded or _load_vlm(model_id)
    grab, close = _grabber(video_path)

    for seg in res['segments']:
        span = seg['end_frame'] - seg['start_frame'] + 1
        idxs = np.linspace(seg['start_frame'], seg['end_frame'],
                           min(frames_per_seg, span)).round().astype(int)
        frames = [f for f in (grab(int(i)) for i in idxs) if f is not None]
        if not frames:
            continue
        seg['open_text'] = _ask(model, processor, frames,
                                OPEN_SEG_PROMPT.format(n=len(frames)), max_new_tokens=24)
        print(f'  段 {seg["start_frame"]:3d}-{seg["end_frame"]:3d}  '
              f'开放描述: {seg["open_text"]}', flush=True)

    N = res['n']
    idxs = np.linspace(0, max(N - 1, 0), min(whole_frames, max(N, 1))).round().astype(int)
    frames = [f for f in (grab(int(i)) for i in idxs) if f is not None]
    task_suggest = None
    if frames:
        task_suggest = _ask(model, processor, frames,
                            OPEN_TASK_PROMPT.format(n=len(frames)), max_new_tokens=32)
    res['open'] = {'model': mpath, 'task_suggest': task_suggest}
    print(f'  [开放识别] 任务概括: {task_suggest}', flush=True)

    close()
    if own:
        _unload_vlm(model)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--export', default=None, help='das_export/<rec>（IMU 定重力轴）')
    ap.add_argument('--out', default=None, help='标注 json 输出（默认 parquet 同名 .actions.json）')
    ap.add_argument('--min-frames', type=int, default=5)
    ap.add_argument('--quantile', action='store_true',
                    help='阈值改按本段分布自适应（默认用物理绝对值 1.05/1.50）')
    ap.add_argument('--write-parquet', action='store_true',
                    help='把逐帧标签写回 parquet 的 action_label 列')
    ap.add_argument('--video', default=None,
                    help='与 parquet 帧号对应的视频（默认自动找 out/rig_input/ 剪切版）')
    ap.add_argument('--vlm', choices=['qwen'], default=None,
                    help='qwen: 规则标注后用 Qwen2.5-VL 抽帧复核各动作段（四类封闭）')
    ap.add_argument('--vlm-open', action='store_true',
                    help='开放动作识别：每段 + 全窗让 Qwen2.5-VL 自由文本描述')
    ap.add_argument('--vlm-model', default='Qwen/Qwen2.5-VL-7B-Instruct-AWQ')
    ap.add_argument('--vlm-frames', type=int, default=3, help='每段抽帧数')
    args = ap.parse_args()

    pq_path = Path(args.parquet)
    res = annotate(pq_path, Path(args.export) if args.export else None,
                   args.min_frames, quantile=args.quantile)

    if args.vlm or args.vlm_open:
        video = Path(args.video) if args.video else find_video(pq_path)
        if video is None or not video.exists():
            raise SystemExit('[vlm] 找不到与 parquet 帧号对应的视频，'
                             f'请用 --video 指定（如 <out>/rig_input/camera0_rig_*.mp4）')
        loaded = None
        try:
            if args.vlm:
                print(f'[vlm] 复核 {len(res["segments"])} 段，抽帧视频: {video.name}',
                      flush=True)
                loaded = _load_vlm(args.vlm_model)
                res = vlm_review(res, video, args.vlm_model, args.vlm_frames,
                                 _loaded=loaded)
            if args.vlm_open:
                print(f'[vlm-open] 开放动作识别（{len(res["segments"])} 段 + 全窗概括）'
                      f'，视频: {video.name}', flush=True)
                if loaded is None:
                    loaded = _load_vlm(args.vlm_model)
                res = vlm_open(res, video, args.vlm_model, _loaded=loaded)
        finally:
            if loaded is not None:
                _unload_vlm(loaded[0])
    out = Path(args.out) if args.out else pq_path.with_suffix('.actions.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.write_parquet:
        df = pq.read_table(pq_path).to_pandas()
        df['action_label'] = res['labels']
        import pyarrow as pa
        tbl = pa.Table.from_pandas(df, schema=pq.read_table(pq_path).schema
                                   .append(pa.field('action_label', pa.string())))
        pq.write_table(tbl, pq_path)

    from collections import Counter
    c = Counter(res['labels'])
    print(f'[actions] {pq_path.name}: {res["n"]} 帧')
    print(f'  阈值: 张开度 p25/p75 = {res["stats"]["open_p25"]:.2f}/{res["stats"]["open_p75"]:.2f}'
          f'  (范围 {res["stats"]["open_min"]:.2f}~{res["stats"]["open_max"]:.2f}),'
          f' 垂直速度阈值 {res["stats"]["vz_thresh_mps"]:.3f} m/s')
    print('  逐帧分布:', {CN[k]: v for k, v in c.most_common()})
    for s in res['segments']:
        print(f'  段 {s["start_frame"]:3d}-{s["end_frame"]:3d}  {s["label_cn"]}')
    if res.get('vlm'):
        v = res['vlm']
        print(f'  [VLM 复核] {v["n_agree"]}/{v["n_segs"]} 段一致'
              f'（一致率 {v["agreement"]:.0%}）' if v['n_segs'] else '  [VLM 复核] 无动作段')
    if res.get('open', {}).get('task_suggest'):
        print(f'  [开放识别] 任务概括: {res["open"]["task_suggest"]}')
    print(f'[OK] {out}')


if __name__ == '__main__':
    main()
