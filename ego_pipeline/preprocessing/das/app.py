# -*- coding: utf-8 -*-
"""DAS 多目重建 本地可视化界面（Flask 单文件）。

启动:
    python ego_pipeline/preprocessing/das/app.py
    # 浏览器打开 http://localhost:7861

功能（对应 CLI 三开关 run_multiview.py --mode）:
    ① 选录制 mcap（下拉，自动扫描数据目录）
    ② 选模式: 单目 / 双目 / 六目
    ③ 设窗口起点/时长 → 运行
    ④ 实时日志 + 完成后页面内播放 multiview.mp4（每次结果独立目录，不覆盖）
"""
from __future__ import annotations

import glob
import html
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

REPO = Path(__file__).resolve().parents[3]   # /home/jun/Ego2LeRobot
PY = Path('/home/jun/miniconda3/envs/ego2lerobot/bin/python')
RUNNER = Path(__file__).resolve().parent / 'run_multiview.py'
OUT_ROOT = REPO / 'examples' / 'das_app_runs'
MCAP_GLOBS = [
    '/mnt/d/work/ego/$person.1/数采集——简智/**/record_data/**/*.mcap',
    '/mnt/d/work/ego/LYZ/**/*.mcap',
]

app = Flask(__name__)
_job = {'proc': None, 'log': [], 'mode': None, 'out': None, 'done': False,
        'video': None, 'error': None, 'started': None}
_lock = threading.Lock()


def list_mcaps() -> list[tuple[str, str]]:
    """[(显示名, 路径)]。DAS 原始 mcap（新在前，>30MB）+ 已转换导出目录（EgoCap 等）。"""
    files: list[Path] = []
    for pat in MCAP_GLOBS:
        files += [Path(f) for f in glob.glob(pat, recursive=True)]
    files = [f for f in files
             if f.stat().st_size > 30 * 1024 * 1024 and 'LYZ' not in f.parts]  # EgoCap 原始 mcap 非DAS格式，走"已转换"入口
    rows = [(f'DAS · {f.name}  ({f.stat().st_size/1e6:.0f}MB)', str(f))
            for f in sorted(files, key=lambda p: p.name, reverse=True)]
    for exp in sorted((REPO / 'examples' / 'egocap_das_export').glob('*/camera0.mp4')):
        rows.append((f'EgoCap已转换 · {exp.parent.name}/', str(exp.parent)))
    return rows


def list_videos() -> list[tuple[str, str]]:
    """历史结果视频（multiview / imu_enhanced），新在前。"""
    pats = ['examples/*/viz/*/multiview.mp4', 'examples/*/*/viz/*/multiview.mp4',
            'examples/*/imu/imu_enhanced.mp4', 'examples/*_imu/imu_enhanced.mp4']
    vids = []
    for pat in pats:
        vids += [Path(p) for p in glob.glob(str(REPO / pat), recursive=True)]
    out = []
    for v in sorted(set(vids), key=lambda p: p.stat().st_mtime, reverse=True)[:60]:
        t = time.strftime('%m-%d %H:%M', time.localtime(v.stat().st_mtime))
        out.append((f'{v.parent.parent.name}/{v.parent.name}  ({t})', str(v)))
    return out


def _watch(proc, out_dir, mode):
    video = None
    for _ in range(60 * 60 * 4):               # 最多 4h
        if proc.poll() is not None:
            break
        time.sleep(1)
    # 找该模式最新时间戳目录里的视频
    cands = sorted((out_dir / 'viz').glob(f'{mode}_*/multiview.mp4'))
    if cands:
        video = str(cands[-1])
    with _lock:
        _job.update(proc=None, done=True, video=video,
                    error=None if proc.returncode == 0 else f'exit={proc.returncode}')
        _job['log'].append(f'\n===== 结束 {"成功" if proc.returncode == 0 else "失败"} '
                           f'{"| 视频: " + video if video else ""}')


@app.get('/')
def index():
    mcaps = list_mcaps()
    vids = list_videos()
    opts = '\n'.join(
        f'<option value="{html.escape(p)}" {"selected" if i == 0 else ""}>{html.escape(n)}</option>'
        for i, (n, p) in enumerate(mcaps))
    vopts = '\n'.join(f'<option value="{html.escape(p)}">{html.escape(n)}</option>'
                      for n, p in vids) or '<option value="">（暂无历史结果）</option>'
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>DAS 多目重建</title><style>
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#10161c;color:#dfe7ee;margin:0;padding:28px}}
 .card{{background:#18222b;border:1px solid #243240;border-radius:12px;padding:20px;max-width:900px;margin:0 auto}}
 h1{{font-size:20px;margin:0 0 6px}} .sub{{color:#8fa3b5;font-size:13px;margin-bottom:18px}}
 .modes{{display:flex;gap:12px;margin:14px 0}}
 .mode{{flex:1;border:2px solid #2c3d4d;border-radius:10px;padding:14px;cursor:pointer;text-align:center;background:#1b2833}}
 .mode b{{display:block;font-size:16px;margin-bottom:4px}} .mode span{{font-size:12px;color:#8fa3b5}}
 .mode.on{{border-color:#37b3ff;background:#14324a}} .mode.on b{{color:#7cc9ff}}
 input[type=number],select{{background:#0d141b;color:#dfe7ee;border:1px solid #2c3d4d;border-radius:8px;padding:8px;font-size:14px}}
 select{{width:100%}} .row{{display:flex;gap:16px;margin:12px 0;align-items:center}}
 button#run{{background:#1f7fd4;color:#fff;border:0;border-radius:8px;padding:12px 34px;font-size:15px;cursor:pointer}}
 button#run:disabled{{background:#33526b;cursor:wait}}
 pre#log{{background:#0a0f14;border:1px solid #1d2933;border-radius:8px;padding:12px;height:220px;overflow:auto;font-size:12px;white-space:pre-wrap}}
 video{{width:100%;border-radius:10px;margin-top:12px;background:#000}}
 .err{{color:#ff8d8d}} .ok{{color:#7ce29b}}
</style></head><body><div class="card">
<h1>DAS 多目重建 · 双手空间轨迹</h1>
<div class="sub">mcap → 去畸变 → 多目重建 → IMU 增强(重力对齐+去抖) → N 输入面板 + 世界系 3D；每次结果独立目录，不覆盖</div>
<div class="modes">
 <div class="mode" data-m="mono"><b>单目</b><span>camera0 · 1 输入 + 世界3D</span></div>
 <div class="mode" data-m="stereo"><b>双目</b><span>camera0+1 · 2 输入 + 世界3D</span></div>
 <div class="mode on" data-m="rig6"><b>六目</b><span>全部 6 目 · 6 输入 + 世界3D</span></div>
</div>
<select id="mcap">{opts}</select>
<div class="row">任务描述(语义标注) <input type="text" id="task" value="" placeholder="留空=默认描述；如: 双手配合拧螺丝组装工件" style="flex:1"></div>
<div class="row">窗口起点(秒) <input type="number" id="start" value="2.4" step="0.1" min="0">
 时长(秒) <input type="number" id="dur" value="5" step="0.5" min="1" max="60">
 <button id="run" onclick="run()">运行</button></div>
<pre id="log">等待运行…（六目约 25–30 分钟：6 目各跑一遍重建管线 + 融合；已有中间产物会自动跳过，重跑很快）</pre>
<div class="row" style="margin-top:14px">历史结果
 <select id="hist" style="flex:1">{vopts}</select>
 <button onclick="playHist()" style="background:#2c3d4d;color:#dfe7ee;border:0;border-radius:8px;padding:10px 22px;cursor:pointer">播放</button></div>
<video id="vid" controls style="display:none"></video>
</div><script>
let mode='rig6';
document.querySelectorAll('.mode').forEach(e=>e.onclick=()=>{{
 document.querySelectorAll('.mode').forEach(x=>x.classList.remove('on'));e.classList.add('on');mode=e.dataset.m;}});
async function playHist(){{
 const v=document.getElementById('vid');
 const p=document.getElementById('hist').value; if(!p)return;
 v.src='/video?p='+encodeURIComponent(p); v.style.display='block'; v.play(); v.scrollIntoView({{behavior:'smooth'}});}}
async function run(){{
 const body={{mode, mcap:document.getElementById('mcap').value,
  start:parseFloat(document.getElementById('start').value),
  duration:parseFloat(document.getElementById('dur').value),
  task:document.getElementById('task').value.trim()}};
 const r=await fetch('/api/run',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
 const j=await r.json(); if(!j.ok){{alert(j.error);return;}}
 document.getElementById('run').disabled=true; document.getElementById('vid').style.display='none';
 poll();}}
async function poll(){{
 const j=await (await fetch('/api/status')).json();
 document.getElementById('log').textContent=j.log;
 if(!j.done){{setTimeout(poll,2000);return;}}
 document.getElementById('run').disabled=false;
 const lg=document.getElementById('log');
 if(j.error){{lg.innerHTML='<span class="err">运行失败：'+j.error+'（详见上方日志）</span>';}}
 if(j.video){{const v=document.getElementById('vid');v.src='/video?p='+encodeURIComponent(j.video);v.style.display='block';v.scrollIntoView({{behavior:'smooth'}});}}}}
</script></body></html>'''


@app.post('/api/run')
def api_run():
    with _lock:
        if _job['proc'] is not None:
            return jsonify(ok=False, error='已有任务在运行')
        req = request.get_json()
        mode, mcap = req.get('mode'), req.get('mcap')
        start, duration = float(req.get('start', 2.4)), float(req.get('duration', 5))
        task = (req.get('task') or '').strip() or None
        if mode not in ('mono', 'stereo', 'rig6') or not Path(mcap).exists():
            return jsonify(ok=False, error='参数无效')
        # 预检时长：mcap 用其本身，已转换目录用 camera0.mp4
        probe_target = str(Path(mcap) / 'camera0.mp4') if not mcap.endswith('.mcap') else mcap
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=duration', '-of', 'csv=p=0', probe_target],
            capture_output=True, text=True)
        try:
            total = float(probe.stdout.strip())
        except ValueError:
            total = 0.0
        if total and start + duration > total * 0.95:
            return jsonify(ok=False,
                           error=f'窗口({start}~{start + duration}s)超出该录制时长({total:.1f}s)，请调小起点/时长')
        src_arg = ['--mcap', mcap] if mcap.endswith('.mcap') else ['--input-export', mcap]
        out = OUT_ROOT / f"{Path(mcap).stem[:40]}_{mode}" if mcap.endswith('.mcap') \
            else OUT_ROOT / f"egocap_{Path(mcap).name}_{mode}"
        tag = uuid.uuid4().hex[:6]
        _job.update(proc=None, log=[], mode=mode, out=out, done=False,
                    video=None, error=None, started=tag)
        _job['log'].append(f'模式={mode}  起点={start}s  时长={duration}s'
                           f'{f"  标注={task}" if task else ""}\n'
                           f'录制={Path(mcap).name}\n输出={out}\n')
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()),
               'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
               'EGO2LEROBOT_GPU_ALLOC_GB': '8',
               'EGO2LEROBOT_RAY_TMP': '/tmp/ray_ego',
               'TORCH_HOME': str(Path.home() / '.cache/torch'),
               'TRITON_CACHE_DIR': '/tmp/triton',
               'PYTHONUNBUFFERED': '1'}
        cmd = [str(PY), str(RUNNER), '--mode', mode, *src_arg,
               '--start', str(start), '--duration', str(duration), '--out', str(out)]
        if task:
            cmd += ['--task', task]
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)

        def pump():
            for line in proc.stdout:
                with _lock:
                    _job['log'].append(line.rstrip())
                    _job['log'] = _job['log'][-400:]
        threading.Thread(target=pump, daemon=True).start()
        threading.Thread(target=_watch, args=(proc, out, mode), daemon=True).start()
        _job['proc'] = proc
        return jsonify(ok=True)


@app.get('/api/status')
def api_status():
    with _lock:
        return jsonify(done=_job['done'], video=_job['video'],
                       error=_job['error'], log='\n'.join(_job['log'][-120:]))


@app.get('/api/runs')
def api_runs():
    return jsonify(videos=[p for _, p in list_videos()])


@app.get('/video')
def video():
    p = request.args.get('p', '')
    if p.endswith('.mp4') and Path(p).exists():
        return send_file(p, mimetype='video/mp4')
    return 'not found', 404


if __name__ == '__main__':
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=7861, threaded=True)
