#!/usr/bin/env python3
"""校验 filtered/ops/hand_filter 的输出是否完整、合理

期望输出结构（与 ops/hand_filter.process_videos 一致）：
    <output_root>/<rel>/<stem>/
        <stem>_part000_<start>s_<end>s.mp4   # 有保留段时
        <stem>_part001_<start>s_<end>s.mp4
        ...
        <stem>.done                         # 跑完必写；整段切光也建目录+写此标记
    <output_root>/.completed.jsonl          # 集中清单；无 .done 时也可当作完成证据

校验项：
    1. 已完成的源：有 <stem>/ 目录，或 jsonl 记录 parts=0（整段被切光）
    2. 完成标记：<stem>.done 或 .completed.jsonl 命中
    3. 有 part 时：每段都能被 ffprobe 解出非零 duration
    4. 有 part 时：文件名里的 [start, end] 互不重叠，总时长 ≤ 源时长 + 容差
    5. （可选 --check_orphan）扫描 output 下多余视频文件

整段切光（parts=0、目录空）是合法结果，不算失败。

用法:
    python verify.py --input /data/src --output /data/src_hand_filter

退出码: 0 全通过；1 存在不一致
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import VIDEO_EXTS, collect_videos

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# 段文件命名：<stem>_partNNN_<start>s_<end>s.mp4  其中 start/end 可能含小数点
_PART_RE = re.compile(
    r"^(?P<stem>.+)_part(?P<idx>\d+)_(?P<start>[0-9]+(?:\.[0-9]+)?)s_(?P<end>[0-9]+(?:\.[0-9]+)?)s$"
)


# ── ffprobe ───────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    """返回 {duration, fps, width, height}；失败返回 {error}"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height:format=duration",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return {"path": path, "error": "ffprobe 未安装"}
    except subprocess.TimeoutExpired:
        return {"path": path, "error": "ffprobe 超时"}
    if r.returncode != 0:
        return {"path": path, "error": r.stderr.strip()[-120:]}
    try:
        data = json.loads(r.stdout)
    except Exception:
        return {"path": path, "error": "json parse"}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    raw = stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, raw.split("/"))
        fps = num / den if den else 0.0
    except Exception:
        fps = 0.0
    duration = float(fmt.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    return {"path": path, "duration": duration, "fps": fps,
            "width": width, "height": height}


def probe_many(paths: list[Path], workers: int, desc: str) -> dict[Path, dict]:
    out: dict[Path, dict] = {}
    if not paths:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe, p): p for p in paths}
        it = as_completed(futs)
        if tqdm:
            it = tqdm(it, total=len(futs), desc=desc, unit="个",
                      dynamic_ncols=True)
        for fut in it:
            r = fut.result()
            out[r["path"]] = r
    return out


# ── 单视频校验 ────────────────────────────────────────────────────────────────

def _parse_part_name(stem: str) -> tuple[int, float, float] | None:
    m = _PART_RE.match(stem)
    if not m:
        return None
    try:
        return (int(m.group("idx")), float(m.group("start")), float(m.group("end")))
    except ValueError:
        return None


def _load_completed_manifest(out_root: Path) -> dict[tuple[str, str], int]:
    """读取 .completed.jsonl，返回 {(out_subdir, stem): parts}。"""
    path = out_root / ".completed.jsonl"
    done: dict[tuple[str, str], int] = {}
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sub = str(Path(rec["out_subdir"]).expanduser().resolve())
                done[(sub, rec["stem"])] = int(rec.get("parts", 0))
            except Exception:
                continue
    return done


def verify_one(src: Path, rel: Path, out_root: Path,
               src_meta: dict, seg_metas: dict[Path, dict],
               args, completed: dict[tuple[str, str], int]) -> dict:
    """返回 {status, src, msg, ...}

    status:
        ok            通过（含整段切光、0 part）
        probe_src     源 ffprobe 失败
        miss_dir      段文件夹不存在且未在 jsonl 记为 parts=0
        miss_done     .done / jsonl 都没有完成记录
        bad_name      part 命名不符 _part<idx>_<start>s_<end>s
        probe_out     段 ffprobe 失败
        overlap       段区间互相重叠
        over_total    段总时长 > 源时长 + 容差
    """
    if "error" in src_meta:
        return {"status": "probe_src", "src": src,
                "msg": f"源 ffprobe 失败: {src_meta['error']}"}
    total = src_meta["duration"]
    if total <= 0:
        return {"status": "probe_src", "src": src, "msg": "源时长 ≤ 0"}

    out_subdir = out_root if str(rel) in ("", ".") else out_root / rel
    seg_folder = out_subdir / src.stem
    key = (str(out_subdir.resolve()), src.stem)
    in_manifest = key in completed
    n_parts_rec = completed.get(key)
    done_marker = seg_folder / f"{src.stem}.done"
    is_done = done_marker.exists() or in_manifest

    if not seg_folder.is_dir():
        if is_done and (n_parts_rec == 0):
            return {"status": "ok", "src": src, "count": 0,
                    "sum_dur": 0.0, "out_paths": []}
        return {"status": "miss_dir", "src": src,
                "msg": f"段文件夹不存在: {seg_folder}"}

    if not is_done:
        return {"status": "miss_done", "src": src,
                "msg": f".done / jsonl 均无完成记录: {done_marker}"}

    parts = sorted(p for p in seg_folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not parts:
        return {"status": "ok", "src": src, "count": 0,
                "sum_dur": 0.0, "out_paths": []}

    # 解析命名
    parsed: list[tuple[Path, int, float, float]] = []
    for p in parts:
        tup = _parse_part_name(p.stem)
        if tup is None:
            return {"status": "bad_name", "src": src,
                    "msg": f"段命名不符规范: {p.name}"}
        parsed.append((p, *tup))

    # 检查 ffprobe
    for p, _idx, _s, _e in parsed:
        m = seg_metas.get(p, {"error": "未 probe"})
        if "error" in m:
            return {"status": "probe_out", "src": src,
                    "msg": f"段 ffprobe 失败 ({p.name}): {m['error']}"}
        if m.get("duration", 0) <= 0:
            return {"status": "probe_out", "src": src,
                    "msg": f"段时长 ≤ 0 ({p.name})"}

    # 检查区间不重叠（按 start 排序）
    parsed_sorted = sorted(parsed, key=lambda x: x[2])
    for (_, _, s1, e1), (_, _, s2, _e2) in zip(parsed_sorted, parsed_sorted[1:]):
        if s2 < e1 - 1e-3:
            return {"status": "overlap", "src": src,
                    "msg": f"段区间重叠: [..., {e1:.2f}s] vs [{s2:.2f}s, ...]"}

    # 检查总长度
    sum_dur = sum(e - s for _, _, s, e in parsed)
    if sum_dur > total + args.dur_tol:
        return {"status": "over_total", "src": src,
                "msg": f"段总时长 {sum_dur:.2f}s > 源时长 {total:.2f}s "
                       f"(容差±{args.dur_tol:.2f}s)"}

    return {"status": "ok", "src": src, "count": len(parsed),
            "sum_dur": sum_dur, "out_paths": [t[0] for t in parsed]}


# ── 多余输出扫描 ─────────────────────────────────────────────────────────────

def find_orphans(out_root: Path, expected_files: set[Path]) -> list[Path]:
    """output_root 下不属于任何已知源的视频文件"""
    orphans = []
    for p in out_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            if p not in expected_files:
                orphans.append(p)
    return orphans


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="源视频目录 / 文件（与 main.py --input 一致）")
    ap.add_argument("--output", required=True,
                    help="待校验的输出根目录（main.py --output_dir 或推断后的根）")
    ap.add_argument("--dur_tol", type=float, default=1.0,
                    help="段总时长相对源时长的容差（秒），默认 1.0")
    ap.add_argument("--workers", type=int, default=16,
                    help="并发 probe 线程数，默认 16")
    ap.add_argument("--show_ok", action="store_true",
                    help="同时打印通过的条目（默认只打印不一致）")
    ap.add_argument("--check_orphan", action="store_true",
                    help="额外扫描 output 下多余的视频文件（源里没有对应）")
    args = ap.parse_args()

    out_root = Path(args.output).expanduser().resolve()
    if not out_root.is_dir():
        print(f"error: --output 目录不存在: {out_root}", file=sys.stderr)
        sys.exit(2)

    videos = collect_videos(args.input)
    if not videos:
        print("error: 源目录未找到视频", file=sys.stderr)
        sys.exit(2)

    print(f"源    : {len(videos)} 个视频", file=sys.stderr)
    print(f"输出根: {out_root}", file=sys.stderr)
    print(f"参数  : dur_tol±{args.dur_tol}s", file=sys.stderr)

    completed = _load_completed_manifest(out_root)
    print(f"jsonl : {len(completed)} 条完成记录", file=sys.stderr)

    # 第一遍 probe 源
    src_metas = probe_many([v for v, _ in videos], args.workers, "probe 源")

    # 推断每个源对应的输出段集合
    output_paths: list[Path] = []
    src_to_outs: dict[Path, list[Path]] = {}
    for v, rel in videos:
        out_subdir = out_root if str(rel) in ("", ".") else out_root / rel
        seg_folder = out_subdir / v.stem
        outs = sorted(p for p in seg_folder.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                      ) if seg_folder.is_dir() else []
        src_to_outs[v] = outs
        output_paths.extend(outs)

    seg_metas = probe_many(output_paths, args.workers, "probe 输出")

    # 校验
    issues: list[dict] = []
    oks: list[dict] = []
    for v, rel in videos:
        r = verify_one(v, rel, out_root, src_metas.get(v, {}), seg_metas, args,
                       completed)
        if r["status"] == "ok":
            oks.append(r)
        else:
            issues.append(r)

    # 输出
    if args.show_ok:
        for r in oks:
            print(f"[ok]    {r['src']}  {r['count']} 段  "
                  f"({r['sum_dur']:.2f}s)")
    for r in issues:
        print(f"[{r['status']:<10}] {r['src']}  {r['msg']}")

    orphans: list[Path] = []
    if args.check_orphan:
        expected = {p for ps in src_to_outs.values() for p in ps}
        orphans = find_orphans(out_root, expected)
        for p in orphans:
            print(f"[orphan    ] {p}")

    # 汇总
    by_status: dict[str, int] = {}
    for r in issues:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    print()
    print("=" * 60)
    print(f"源视频    : {len(videos)}")
    print(f"通过      : {len(oks)}")
    if issues:
        print(f"不一致    : {len(issues)}")
        for st, n in sorted(by_status.items(), key=lambda x: -x[1]):
            print(f"  {st:<10}: {n}")
    if args.check_orphan:
        print(f"多余文件  : {len(orphans)}")

    if issues or orphans:
        sys.exit(1)


if __name__ == "__main__":
    main()
