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
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

REPO = Path(__file__).resolve().parents[3]   # repository root
PY = Path(os.environ.get('EGO2LEROBOT_PY') or sys.executable)
RUNNER = Path(__file__).resolve().parent / 'run_multiview.py'
OUT_ROOT = REPO / 'examples' / 'das_app_runs'
# Semicolon-separated glob patterns, e.g.
# EGO2LEROBOT_MCAP_GLOBS='/data/ego/**/*.mcap;/mnt/d/ego/**/*.mcap'
MCAP_GLOBS = [
    pat for pat in os.environ.get('EGO2LEROBOT_MCAP_GLOBS', '').split(';') if pat
]

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    return resp


_job = {'proc': None, 'log': [], 'mode': None, 'out': None, 'done': False,
        'video': None, 'error': None, 'started': None}
_imp = {'busy': False, 'log': [], 'done': True, 'ok': False, 'error': None,
        'name': None}
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


_rev = {'proc': None, 'url': None}


def _launch_review(out_dir, export):
    """为本次运行拉起人工复核界面（端口 7864，已运行则复用）。"""
    if _rev['proc'] is not None and _rev['proc'].poll() is None:
        return _rev['url']
    import subprocess as _sp
    _rev['proc'] = _sp.Popen(
        [str(PY), str(HERE / 'hoi_review_app.py'),
         '--structured-dir', str(out_dir / 'structured'),
         '--export', str(export), '--port', '7864'],
        cwd=str(REPO), stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    _rev['url'] = 'http://localhost:7864'
    return _rev['url']


def _watch(proc, out_dir, mode):
    video = None
    for _ in range(60 * 60 * 4):               # 最多 4h
        if proc.poll() is not None:
            break
        time.sleep(1)
    # 找该模式最新时间戳目录里的视频
    cands = sorted((out_dir / 'viz').glob(f'{mode}*/multiview.mp4'),
                   key=lambda p: ('+hoi' not in str(p), str(p)))  # 匹配含+hoi版且优先
    review = sorted((out_dir / 'structured/hoi_fusion/review_videos').glob('*_review.mp4'))
    if review:
        _job['videos'] = [str(p) for p in review]
        _job['review_url'] = _launch_review(out_dir, _job.get('last_export') or out_dir)
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
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#10161c;color:#dfe7ee;margin:0;padding:18px}}
 .card{{background:#18222b;border:1px solid #243240;border-radius:12px;padding:22px;width:97vw;max-width:1720px;margin:0 auto}}
 h1{{font-size:22px;margin:0 0 6px}} .sub{{color:#8fa3b5;font-size:13px;margin-bottom:14px}}
 .grid{{display:grid;grid-template-columns:420px 1fr;gap:20px;margin-top:12px}}
 .modes{{display:flex;gap:10px;margin:10px 0}}
 .mode{{flex:1;border:2px solid #2c3d4d;border-radius:10px;padding:12px 8px;cursor:pointer;text-align:center;background:#1b2833}}
 .mode b{{display:block;font-size:16px;margin-bottom:4px}} .mode span{{font-size:12px;color:#8fa3b5}}
 .mode.on{{border-color:#37b3ff;background:#14324a}} .mode.on b{{color:#7cc9ff}}
 input[type=number],input[type=text],select{{background:#0d141b;color:#dfe7ee;border:1px solid #2c3d4d;border-radius:8px;padding:8px;font-size:14px}}
 select{{width:100%}} .row{{display:flex;gap:12px;margin:10px 0;align-items:center;flex-wrap:wrap}}
 button#run{{background:#1f7fd4;color:#fff;border:0;border-radius:8px;padding:12px 34px;font-size:15px;cursor:pointer}}
 button#run:disabled{{background:#33526b;cursor:wait}}
 #errmsg{{color:#ff8d8d;background:#3a1520;border:1px solid #7c2b3a;border-radius:8px;padding:8px 12px;display:none;margin:0 0 8px;white-space:pre-wrap;font-size:13px}}
 pre#log{{background:#0a0f14;border:1px solid #1d2933;border-radius:8px;padding:12px;height:60vh;min-height:360px;overflow:auto;font-size:12.5px;white-space:pre-wrap;margin:0}}
 video{{width:100%;border-radius:10px;margin-top:12px;background:#000}}
 .lbl{{font-size:12px;color:#8fa3b5;margin:6px 0 4px}}
</style></head><body><div class="card">
<h1>DAS 多目重建 · 双手空间轨迹</h1>
<div class="sub">mcap → 去畸变 → 多目重建 → IMU 增强(重力对齐+去抖) → N 输入面板 + 世界系 3D；每次结果独立目录，不覆盖</div>
<div class="grid">
<div class="left">
 <div class="modes">
  <div class="mode" data-m="mono"><b>单目</b><span>camera0 · 1 输入</span></div>
  <div class="mode" data-m="stereo"><b>双目</b><span>camera0+1 · 2 输入</span></div>
  <div class="mode on" data-m="rig6"><b>六目</b><span>全部 6 目</span></div>
 </div>
 <div class="lbl">数据源（DAS 原始 mcap / EgoCap 已转换目录）</div>
 <select id="mcap">{opts}</select>
 <div class="lbl" style="margin-top:12px">导入本地 EgoCap 数据（同事的 bag，转好即出现在上方下拉）</div>
 <div class="row"><input type="text" id="import_path" placeholder="/mnt/d/work/ego/LYZ/egocap_xxx/bag 或 bag_X.mcap" style="flex:1;min-width:150px">
  <button id="impbtn" onclick="doImport()" style="background:#2c7a4b;color:#fff;border:0;border-radius:8px;padding:10px 18px;cursor:pointer">导入并转换</button></div>
 <div id="impstat" style="font-size:12px;color:#8fa3b5;white-space:pre-wrap"></div>
 <div class="row">任务描述(语义标注) <input type="text" id="task" value="" placeholder="留空=默认；如: 双手配合拧螺丝组装工件" style="flex:1;min-width:140px"></div>
 <div class="row">窗口起点(秒) <input type="number" id="start" value="2.4" step="0.1" min="0" style="width:90px">
  时长(秒) <input type="number" id="dur" value="5" step="0.5" min="1" max="60" style="width:90px"></div>
 <div class="row"><button id="run" onclick="run()">运行</button>
  <label style="font-size:12px;color:#8fa3b5"><input type="checkbox" id="audit_fill" checked> 掉帧检查+补帧</label>
  <label style="font-size:12px;color:#8fa3b5"><input type="checkbox" id="hoi" checked> 语义标注视频(中文红框,含大模型约+3分钟)</label>
  <label style="font-size:12px;color:#8fa3b5"><input type="checkbox" id="vlm"> 大模型复核动作</label>
  <label style="font-size:12px;color:#8fa3b5"><input type="checkbox" id="vlm_open"> 开放动作识别(自动任务描述)</label>
  <label style="font-size:12px;color:#8fa3b5"><input type="checkbox" id="autoscroll" checked> 日志自动滚动</label></div>
 <div class="lbl">历史结果（选一条点播放，随时回看）</div>
 <div class="row"><select id="hist" style="flex:1">{vopts}</select>
  <button onclick="playHist()" style="background:#2c3d4d;color:#dfe7ee;border:0;border-radius:8px;padding:10px 22px;cursor:pointer">播放</button></div>
 <div class="lbl" style="margin-top:14px">耗时参考：单目≈3-4分 双目≈8分 六目≈25-30分；同源同模式重跑复用缓存，通常&lt;1分钟</div>
</div>
<div class="right">
 <div id="errmsg"></div>
 <pre id="log">等待运行…</pre>
 <video id="vid" controls style="display:none"></video>
</div>
</div></div><script>
let mode='rig6';
document.querySelectorAll('.mode').forEach(e=>e.onclick=()=>{{
 document.querySelectorAll('.mode').forEach(x=>x.classList.remove('on'));e.classList.add('on');mode=e.dataset.m;}});
async function playHist(){{
 const v=document.getElementById('vid');
 const p=document.getElementById('hist').value; if(!p)return;
 v.src='/video?p='+encodeURIComponent(p); v.style.display='block'; v.play(); v.scrollIntoView({{behavior:'smooth'}});}}
async function doImport(){{
 const p=document.getElementById('import_path').value.trim(); if(!p){{alert('请填写 bag 目录或 mcap 路径');return;}}
 const r=await fetch('/api/import',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{path:p}})}});
 const j=await r.json(); if(!j.ok){{alert(j.error);return;}}
 document.getElementById('impbtn').disabled=true;
 document.getElementById('impstat').textContent='转换中…';
 pollImport();}}
async function pollImport(){{
 const j=await (await fetch('/api/import/status')).json();
 document.getElementById('impstat').textContent=j.log;
 if(!j.done){{setTimeout(pollImport,2000);return;}}
 document.getElementById('impbtn').disabled=false;
 if(j.ok){{
   const s=await (await fetch('/api/sources')).text();
   document.getElementById('mcap').innerHTML=s;
   document.getElementById('impstat').textContent+='\\n✅ 已加入数据源下拉（拉到 EgoCap已转换 · '+j.name+'/），可直接选运行';
 }} }}
async function run(){{
 const body={{mode, mcap:document.getElementById('mcap').value,
  start:parseFloat(document.getElementById('start').value),
  duration:parseFloat(document.getElementById('dur').value),
  task:document.getElementById('task').value.trim(),
  vlm:document.getElementById('vlm').checked,
  audit_fill:document.getElementById('audit_fill').checked,
  hoi:document.getElementById('hoi').checked,
  vlm_open:document.getElementById('vlm_open').checked}};
 const r=await fetch('/api/run',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
 const j=await r.json(); if(!j.ok){{alert(j.error);return;}}
 document.getElementById('run').disabled=true; document.getElementById('vid').style.display='none';
 document.getElementById('errmsg').style.display='none';
 poll();}}
async function poll(){{
 const j=await (await fetch('/api/status')).json();
 const lg=document.getElementById('log');
 const stick=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-40;   /* 用户是否在底部 */
 if(lg.textContent!==j.log){{
  lg.textContent=j.log;
  if(stick&&document.getElementById('autoscroll').checked) lg.scrollTop=lg.scrollHeight;}}
 const em=document.getElementById('errmsg');
 if(j.error){{em.style.display='block';em.textContent='运行失败：'+j.error+'（完整报错请上滑日志查看）';}}
 else em.style.display='none';
 if(!j.done){{setTimeout(poll,2000);return;}}
 document.getElementById('run').disabled=false;
 if(j.video){{const v=document.getElementById('vid');v.src='/video?p='+encodeURIComponent(j.video);v.style.display='block';}}
 if(j.videos&&j.videos.length){{const d=document.getElementById('rv');d.style.display='block';
   d.innerHTML='<b>语义标注视频(中文红框):</b><br>'+j.videos.map((p,i)=>`<video controls width="640" src="/video?p=${{encodeURIComponent(p)}}"></video>`).join('<br>');}}}}
 if(j.review_url){{document.getElementById('rvlink').style.display='inline';document.getElementById('rvlink').href=j.review_url;}}
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
        vlm = bool(req.get('vlm'))
        vlm_open = bool(req.get('vlm_open'))
        audit_fill = bool(req.get('audit_fill'))
        hoi = bool(req.get('hoi'))
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
        _job['last_export'] = mcap if not mcap.endswith('.mcap') else None
        out = OUT_ROOT / f"{Path(mcap).stem[:40]}_{mode}" if mcap.endswith('.mcap') \
            else OUT_ROOT / f"egocap_{Path(mcap).name}_{mode}"
        tag = uuid.uuid4().hex[:6]
        _job.update(proc=None, log=[], mode=mode, out=out, done=False,
                    video=None, error=None, started=tag)
        _job['log'].append(f'模式={mode}  起点={start}s  时长={duration}s'
                           f'{f"  标注={task}" if task else ""}'
                           f'{"  掉帧检查+补帧=开" if audit_fill else ""}'
                           f'{"  语义标注视频=开" if hoi else ""}'
                           f'{"  VLM复核=开" if vlm else ""}'
                           f'{"  开放识别=开" if vlm_open else ""}\n'
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
        if audit_fill:
            cmd += ['--audit-fill']
        if hoi:
            cmd += ['--hoi']
        if vlm:
            cmd += ['--vlm']
        if vlm_open:
            cmd += ['--vlm-open']
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)

        def pump():
            for line in proc.stdout:
                with _lock:
                    _job['log'].append(line.rstrip())
                    _job['log'] = _job['log'][-3000:]
        threading.Thread(target=pump, daemon=True).start()
        threading.Thread(target=_watch, args=(proc, out, mode), daemon=True).start()
        _job['proc'] = proc
        return jsonify(ok=True)


@app.get('/api/status')
def api_status():
    with _lock:
        return jsonify(done=_job['done'], video=_job['video'],
                       videos=_job.get('videos'),
                       review_url=_job.get('review_url'),
                       error=_job['error'], log='\n'.join(_job['log'][-3000:]))


@app.get('/api/imu')
def api_imu():
    """渲染本窗口的 IMU 曲线 PNG（陀螺/加速度）。DAS 首跑导出前会 404，前端自动重试。"""
    import hashlib

    import numpy as np
    dirp = Path(request.args.get('dir', ''))
    start = float(request.args.get('start', 0))
    dur = float(request.args.get('dur', 5))
    ch = request.args.get('ch', 'gyro')
    f = dirp / 'imu.npz'
    if not f.exists():
        return jsonify(ok=False, error='imu.npz 不存在（可能尚未导出）')
    d = np.load(f)
    t = d['t_log_ns'].astype(np.float64)
    # 窗口零点：优先用 camera0 逐帧时间戳；相对钟（EgoCap）自动平移到 IMU 绝对钟
    tsf = dirp / 'camera0_timestamps.npz'
    if tsf.exists():
        ts0 = float(np.load(tsf)['log_time_ns'][0])
    else:
        ts0 = float(t[0])
    if abs(ts0) < 1e15 and abs(float(t[0])) > 1e15:
        ts0 = float(t[0])                       # 相对时间戳 → 对齐 IMU 起点
    t0 = ts0
    m = (t >= t0 + start * 1e9) & (t < t0 + (start + dur) * 1e9)
    if m.sum() < 5:
        return jsonify(ok=False, error='窗口内无 IMU 采样')
    tt = (t[m] - (t0 + start * 1e9)) / 1e9

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
        plt.rcParams['font.family'] = font_manager.FontProperties(
            fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    if ch == 'gyro':
        w = d['gyro_rad_s'][m] * 180.0 / np.pi
        for j, c in enumerate(('#1f77b4', '#2ca02c', '#d62728')):
            ax.plot(tt, w[:, j], color=c, lw=0.9, label='xyz'[j])
        ax.set_ylabel('°/s')
        ax.set_title(f'陀螺仪 ω（机体系，{dur:.1f}s 窗口，{m.sum()} 采样）')
    else:
        a = d['accel_g'][m] * 9.80665
        for j, c in enumerate(('#1f77b4', '#2ca02c', '#d62728')):
            ax.plot(tt, a[:, j], color=c, lw=0.9, label='xyz'[j])
        ax.axhline(-9.80665, color='0.5', lw=0.8, ls=':')
        ax.set_ylabel('m/s²')
        ax.set_title(f'加速度 a（机体系，含重力 ≈ |a| 9.8）')
    ax.set_xlabel('窗口内秒')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    fig.tight_layout()
    key = hashlib.md5(f'{dirp}{start:.2f}{dur:.2f}{ch}'.encode()).hexdigest()[:10]
    png = Path('/tmp') / f'imu_ui_{key}.png'
    fig.savefig(png, facecolor='white')
    plt.close(fig)
    from urllib.parse import quote
    return jsonify(ok=True, img='/img?p=' + quote(str(png)))


@app.get('/img')
def img():
    p = request.args.get('p', '')
    if p.startswith('/tmp/imu_ui_') and p.endswith('.png') and Path(p).exists():
        return send_file(p, mimetype='image/png')
    return 'not found', 404


@app.get('/api/sources')
def api_sources():
    import html as _h
    return '\n'.join(
        f'<option value="{_h.escape(p)}">{_h.escape(n)}</option>'
        for n, p in list_mcaps())


def _to_wsl_path(p: str) -> str:
    """容忍直接粘贴的 Windows 路径/引号：D:\\x\\y 或 D:/x/y → /mnt/d/x/y。"""
    p = p.strip().strip('"').strip("'")
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m:
        drv, rest = m.groups()
        return f'/mnt/{drv.lower()}/' + rest.replace('\\', '/')
    return p


@app.post('/api/import')
def api_import():
    with _lock:
        if _imp['busy']:
            return jsonify(ok=False, error='已有导入在进行')
        req = request.get_json()
        path = _to_wsl_path(req.get('path') or '')
        p = Path(path)
        if not path or not (p.is_dir() or p.is_file()):
            return jsonify(ok=False, error=f'路径不存在: {path}')
        _imp.update(busy=True, log=[], done=False, ok=False, error=None, name=None)
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()),
               'PYTHONUNBUFFERED': '1'}
        proc = subprocess.Popen(
            [str(PY), str(RUNNER.parent / 'convert_egocap.py'), '--input', path],
            cwd=str(REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _pump():
            for line in proc.stdout:
                with _lock:
                    _imp['log'].append(line.rstrip())
                    _imp['log'] = _imp['log'][-200:]
            proc.wait()
            with _lock:
                _imp.update(busy=False, done=True,
                            ok=proc.returncode == 0,
                            error=None if proc.returncode == 0
                            else f'exit={proc.returncode}')
                if proc.returncode == 0:
                    for line in reversed(_imp['log']):
                        if line.startswith('[convert] 导出名:'):
                            _imp['name'] = line.split(':', 1)[1].strip()
                            break
        threading.Thread(target=_pump, daemon=True).start()
        return jsonify(ok=True)


@app.get('/api/import/status')
def api_import_status():
    with _lock:
        return jsonify(done=_imp['done'], ok=_imp['ok'],
                       error=_imp['error'], name=_imp['name'],
                       log='\n'.join(_imp['log'][-30:]))


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
