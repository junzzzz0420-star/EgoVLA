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
import socket
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_file

from review_workflow import find_run_from_video, infer_context, load_context

REPO = Path(__file__).resolve().parents[3]   # repository root
PY = Path(os.environ.get('EGO2LEROBOT_PY') or sys.executable)
RUNNER = Path(__file__).resolve().parent / 'run_multiview.py'
REVIEW_APP = Path(__file__).resolve().parent / 'hoi_review_app.py'
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
_review = {'proc': None, 'context': None, 'log': []}
_lock = threading.Lock()
_review_lock = threading.Lock()


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


def _start_review_server(context_path: Path) -> None:
    """确保 7863 指向所选运行；切换运行时只重启复核页，不影响主任务。"""
    context_path = context_path.resolve()
    with _review_lock:
        proc = _review.get('proc')
        if (proc is not None and proc.poll() is None
                and _review.get('context') == str(context_path)):
            return
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        proc = subprocess.Popen(
            [str(PY), str(REVIEW_APP), '--context', str(context_path), '--port', '7863',
             '--parent-pid', str(os.getpid())],
            cwd=str(REPO), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        _review.update(proc=proc, context=str(context_path), log=[])

        def pump():
            for line in proc.stdout:
                with _review_lock:
                    _review['log'].append(line.rstrip())
                    _review['log'] = _review['log'][-500:]
        threading.Thread(target=pump, daemon=True).start()

    deadline = time.time() + 12
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('人工复核服务启动失败：' + '\n'.join(_review['log'][-20:]))
        try:
            with socket.create_connection(('127.0.0.1', 7863), timeout=.25):
                return
        except OSError:
            time.sleep(.15)
    raise RuntimeError('人工复核服务启动超时')


def _context_for_video(raw: str) -> tuple[Path, dict]:
    video = Path(raw)
    run = find_run_from_video(video, REPO / 'examples')
    return infer_context(run, video)


def _watch(proc, out_dir, mode):
    video = None
    for _ in range(60 * 60 * 4):               # 最多 4h
        if proc.poll() is not None:
            break
        time.sleep(1)
    # 找该模式最新时间戳目录里的视频
    cands = sorted((out_dir / 'viz').glob(f'{mode}*/multiview.mp4'),
                   key=lambda p: ('+hoi' not in str(p), str(p)))  # 匹配含+hoi版且优先
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAS · EGO 数据工作台</title><style>
:root{{--bg:#080d18;--panel:#111827;--panel2:#151f31;--line:#25324a;--text:#edf3fb;--muted:#8fa0b8;--blue:#4ca9ff;--blue2:#2177e8;--green:#39d39f;--orange:#f2a84b;--danger:#ff6b7a}}
*{{box-sizing:border-box}} body{{font-family:"Microsoft YaHei",Inter,system-ui,sans-serif;background:radial-gradient(circle at 12% 0,#142747 0,transparent 28%),var(--bg);color:var(--text);margin:0;min-height:100vh}}
button,input,select{{font:inherit}} .shell{{width:min(1880px,100%);margin:auto;padding:22px}}
.topbar{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:16px}}
.brand{{display:flex;align-items:center;gap:14px}} .brandmark{{width:46px;height:46px;border-radius:13px;display:grid;place-items:center;font-weight:800;letter-spacing:.06em;background:linear-gradient(145deg,#2898ff,#645bff);box-shadow:0 10px 30px #147ee644}}
.eyebrow{{font-size:11px;letter-spacing:.18em;color:#6fbaff;font-weight:700}} h1{{font-size:22px;margin:3px 0 0;letter-spacing:.01em}} .sub{{color:var(--muted);font-size:12px;margin-top:4px}}
.online{{display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #244b42;background:#102621;border-radius:999px;color:#80e9c4;font-size:12px}} .online i{{width:7px;height:7px;background:var(--green);border-radius:50%;box-shadow:0 0 12px var(--green)}}
.pipeline{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:16px}} .pipe{{position:relative;background:#0d1523;border:1px solid #1c2940;border-radius:10px;padding:9px 11px;color:#9eacc0;font-size:11px}} .pipe b{{display:block;color:#dfe9f7;font-size:12px;margin-bottom:2px}} .pipe::after{{content:'›';position:absolute;right:-8px;top:10px;color:#47617f;z-index:2}} .pipe:last-child::after{{display:none}}
.workspace{{display:grid;grid-template-columns:420px minmax(0,1fr);gap:16px;align-items:start}} .sidebar{{display:grid;gap:12px;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto;padding-right:3px}}
.panel{{background:linear-gradient(160deg,#131d2d,#0f1725);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 40px #0003}}
.control-card{{padding:17px}} .section-head{{display:flex;align-items:center;gap:10px;margin-bottom:14px}} .section-head .num{{display:grid;place-items:center;width:25px;height:25px;border-radius:7px;background:#173a5b;color:#75c2ff;font-weight:700;font-size:11px}} .section-head b{{font-size:14px}} .section-head span:last-child{{margin-left:auto;color:var(--muted);font-size:11px}}
.field{{display:block;margin:10px 0}} .field-name{{display:block;font-size:11px;color:var(--muted);margin:0 0 6px}}
input[type=number],input[type=text],select{{width:100%;color:var(--text);background:#09111e;border:1px solid #293954;border-radius:9px;padding:10px 11px;outline:none;transition:.2s}} input:focus,select:focus{{border-color:var(--blue);box-shadow:0 0 0 3px #278fff20}}
.mode-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .mode{{border:1px solid #2b3950;border-radius:10px;padding:10px 5px;cursor:pointer;text-align:center;background:#0c1523;transition:.2s}} .mode b{{display:block;font-size:13px}} .mode span{{font-size:9px;color:var(--muted)}} .mode:hover{{transform:translateY(-1px);border-color:#46617e}} .mode.on{{border-color:var(--blue);background:linear-gradient(160deg,#12365a,#10233d);box-shadow:inset 0 0 0 1px #4ca9ff55}} .mode.on b{{color:#8dccff}}
.inline{{display:flex;gap:8px;align-items:center}} .inline .grow{{flex:1;min-width:0}} .split{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}
.btn{{border:0;border-radius:9px;padding:10px 14px;color:#fff;cursor:pointer;white-space:nowrap;transition:.18s}} .btn:hover{{filter:brightness(1.1);transform:translateY(-1px)}} .btn-secondary{{background:#24334a}} .btn-import{{background:#176547}} .btn-review{{background:linear-gradient(135deg,#c5721a,#9c5011)}}
#run{{width:100%;border:0;border-radius:10px;padding:12px 18px;background:linear-gradient(135deg,var(--blue2),#645bff);color:white;font-weight:700;cursor:pointer;box-shadow:0 9px 24px #2177e833}} #run:disabled{{background:#304157;box-shadow:none;cursor:wait}}
.toggle-grid{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0}} .toggle{{display:flex;align-items:flex-start;gap:7px;padding:8px;border:1px solid #22314a;border-radius:9px;background:#0b1422;color:#bac7d7;font-size:11px;line-height:1.35}} .toggle input{{accent-color:var(--blue);margin-top:2px}}
.hint{{font-size:10px;color:#73849d;line-height:1.55}} #impstat,#reviewstat{{font-size:11px;color:#83a1bb;white-space:pre-wrap;line-height:1.45}}
.main{{display:grid;gap:12px;min-width:0}} .result-card{{padding:17px}} .result-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px}} .result-head h2{{font-size:16px;margin:0 0 4px}} .result-head p{{font-size:11px;color:var(--muted);margin:0}} .candidate{{font-size:10px;padding:5px 9px;border-radius:999px;color:#ffd296;border:1px solid #694820;background:#2b2114}}
#errmsg{{color:#ffb0b7;background:#35131c;border:1px solid #7c2b3a;border-radius:9px;padding:9px 12px;display:none;margin-bottom:10px;white-space:pre-wrap;font-size:12px}}
.video-stage{{position:relative;display:grid;place-items:center;min-height:440px;border:1px solid #24334a;border-radius:13px;overflow:hidden;background:linear-gradient(135deg,#080d15,#0c1523)}} .empty{{text-align:center;color:#6f829b;padding:50px}} .empty-icon{{width:66px;height:66px;margin:0 auto 14px;border:1px solid #2d4664;border-radius:20px;display:grid;place-items:center;font-size:25px;color:#69b9ff;background:#10243b}} .empty b{{display:block;color:#aebfd2;font-size:14px;margin-bottom:6px}} video{{display:none;width:100%;max-height:72vh;background:#000}}
#main-result{{display:none;margin-top:10px;padding:9px 11px;background:#0c1726;border-radius:8px;color:#8fa9c3;font-size:11px;word-break:break-all}}
.log-card{{overflow:hidden}} .log-card summary{{cursor:pointer;list-style:none;padding:13px 16px;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:space-between}} .log-card summary::-webkit-details-marker{{display:none}} .log-card summary span{{font-weight:400;color:var(--muted)}} pre#log{{background:#070c13;border-top:1px solid #1d2a3e;padding:13px;height:230px;overflow:auto;font:11px/1.55 "Cascadia Mono",Consolas,monospace;white-space:pre-wrap;margin:0;color:#b6c3d2}}
@media(max-width:1050px){{.workspace{{grid-template-columns:1fr}}.sidebar{{position:static;max-height:none;overflow:visible}}.pipeline{{grid-template-columns:repeat(3,1fr)}}.video-stage{{min-height:300px}}}} @media(max-width:620px){{.shell{{padding:12px}}.topbar{{align-items:flex-start}}.online{{display:none}}.pipeline{{grid-template-columns:1fr 1fr}}.workspace{{gap:10px}}.toggle-grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="shell">
<header class="topbar"><div class="brand"><div class="brandmark">DAS</div><div><div class="eyebrow">EGO DATA WORKSPACE</div><h1>双手空间轨迹工作台</h1><div class="sub">多目重建、语义理解与人工复核的一体化流水线</div></div></div><div class="online"><i></i>本地服务运行中</div></header>
<div class="pipeline"><div class="pipe"><b>01 数据输入</b>MCAP / EgoCap</div><div class="pipe"><b>02 几何处理</b>去畸变与同步</div><div class="pipe"><b>03 手部重建</b>多目轨迹融合</div><div class="pipe"><b>04 IMU 增强</b>重力对齐与去抖</div><div class="pipe"><b>05 语义核验</b>VLM + 检测 + 3D</div><div class="pipe"><b>06 人工裁决</b>版本化回写</div></div>
<div class="workspace"><aside class="sidebar">
 <section class="panel control-card"><div class="section-head"><span class="num">01</span><b>数据与视角</b><span>INPUT</span></div>
  <div class="mode-grid"><div class="mode" data-m="mono"><b>单目</b><span>CAMERA 0</span></div><div class="mode" data-m="stereo"><b>双目</b><span>CAMERA 0 + 1</span></div><div class="mode on" data-m="rig6"><b>六目</b><span>FULL RIG</span></div></div>
  <label class="field"><span class="field-name">数据源</span><select id="mcap">{opts}</select></label>
  <label class="field"><span class="field-name">导入本地 EgoCap 数据</span><div class="inline"><input class="grow" type="text" id="import_path" placeholder="bag 目录或 .mcap 路径"><button class="btn btn-import" id="impbtn" onclick="doImport()">导入</button></div></label><div id="impstat"></div>
 </section>
 <section class="panel control-card"><div class="section-head"><span class="num">02</span><b>运行参数</b><span>CONFIG</span></div>
  <label class="field"><span class="field-name">任务描述 / 语义提示</span><input type="text" id="task" value="" placeholder="例如：双手配合组装工件"></label>
  <div class="split"><label class="field"><span class="field-name">窗口起点（秒）</span><input type="number" id="start" value="2.4" step="0.1" min="0"></label><label class="field"><span class="field-name">分析时长（秒）</span><input type="number" id="dur" value="5" step="0.5" min="1" max="60"></label></div>
  <div class="toggle-grid"><label class="toggle"><input type="checkbox" id="audit_fill" checked><span>掉帧检查<br>自动补帧</span></label><label class="toggle"><input type="checkbox" id="hoi" checked><span>红框语义<br>交互标注</span></label><label class="toggle"><input type="checkbox" id="vlm"><span>VLM 动作<br>复核</span></label><label class="toggle"><input type="checkbox" id="vlm_open"><span>开放动作<br>自动描述</span></label></div>
  <button id="run" onclick="run()">开始运行</button><div class="hint" style="margin-top:9px">参考耗时：单目 3–4 分钟 · 双目约 8 分钟 · 六目 25–30 分钟；缓存命中通常少于 1 分钟。</div>
 </section>
 <section class="panel control-card"><div class="section-head"><span class="num">03</span><b>结果与复核</b><span>REVIEW</span></div>
  <label class="field"><span class="field-name">历史运行结果</span><select id="hist">{vopts}</select></label>
  <div class="inline"><button class="btn btn-secondary" onclick="playHist()">播放结果</button><button class="btn btn-review" id="reviewbtn" onclick="openReview()">人工复核</button><label class="toggle" style="margin-left:auto;padding:7px"><input type="checkbox" id="autoscroll" checked><span>日志跟随</span></label></div>
  <div id="reviewstat" style="margin-top:10px">机器候选 → 证据核验 → 人工权威版本</div>
 </section>
</aside><main class="main">
 <section class="panel result-card"><div class="result-head"><div><h2>主运行结果</h2><p>手部建模、世界系轨迹与红色语义标注统一输出</p></div><span class="candidate">机器候选 · 人工审核优先</span></div><div id="errmsg"></div>
  <div class="video-stage"><div class="empty" id="empty"><div class="empty-icon">▶</div><b>等待运行结果</b><span>完成运行或从历史记录中选择视频</span></div><video id="vid" controls></video></div><div id="main-result"></div>
 </section>
 <details class="panel log-card" open><summary>运行日志 <span>实时输出 · 支持折叠</span></summary><pre id="log">等待运行…</pre></details>
</main></div></div><script>
let mode='rig6',reviewPollToken=0;
document.querySelectorAll('.mode').forEach(e=>e.onclick=()=>{{
 document.querySelectorAll('.mode').forEach(x=>x.classList.remove('on'));e.classList.add('on');mode=e.dataset.m;}});
function showVideo(p,label,autoplay=true){{
 const v=document.getElementById('vid');document.getElementById('empty').style.display='none';
 v.dataset.path=p;v.src='/video?p='+encodeURIComponent(p)+'&rev='+Date.now();v.style.display='block';
 const r=document.getElementById('main-result');r.style.display='block';r.textContent=label+'：'+p;
 if(autoplay)v.play().catch(()=>{{}});return v;}}
async function playHist(){{
 const p=document.getElementById('hist').value; if(!p)return;
 const v=showVideo(p,'历史结果');v.scrollIntoView({{behavior:'smooth',block:'center'}});}}
function currentVideoPath(){{
 const v=document.getElementById('vid');
 return v.dataset.path||document.getElementById('hist').value||'';}}
function openReview(){{
 const p=currentVideoPath(); if(!p){{alert('请先运行或选择一条历史结果');return;}}
 document.getElementById('reviewstat').textContent='正在打开复核队列；提交后主视频会自动刷新…';
 window.open('/review/open?video='+encodeURIComponent(p),'hoi_review');
 const token=++reviewPollToken;pollReview(p,null,token);}}
async function pollReview(source,baseVersion,token){{
 if(token!==reviewPollToken)return;
 try{{
  const j=await (await fetch('/api/review/status?video='+encodeURIComponent(source))).json();
  const s=document.getElementById('reviewstat');
  if(!j.ok){{s.textContent='复核状态：'+j.error;return;}}
  s.textContent='复核状态：'+(j.status==='human_reviewed'?'人工版 v'+String(j.review_version).padStart(3,'0'):'待人工核验');
  if(baseVersion===null){{
   baseVersion=j.review_version;
   if(j.status==='human_reviewed'&&j.main_video&&j.main_video!==currentVideoPath())showVideo(j.main_video,'当前人工复核版');
  }}else if(j.review_version>baseVersion&&j.main_video){{
   showVideo(j.main_video,'人工复核版已同步');s.textContent='复核完成：人工版 v'+String(j.review_version).padStart(3,'0');return;
  }}
  setTimeout(()=>pollReview(source,baseVersion,token),5000);
 }}catch(e){{document.getElementById('reviewstat').textContent='复核状态读取失败：'+e;}}
}}
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
 const empty=document.getElementById('empty');empty.style.display='block';
 empty.querySelector('b').textContent='正在生成结果';empty.querySelector('span').textContent='可在下方展开日志查看实时进度';
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
 if(j.video)showVideo(j.video,'主运行视频已保存',false);
 }}
</script></body></html>'''


@app.get('/review/open')
def open_review():
    """从主界面一键启动所选结果的复核页，并在新标签打开。"""
    try:
        context_path, _ = _context_for_video(request.args.get('video', ''))
        _start_review_server(context_path)
    except Exception as exc:
        return (f'<meta charset="utf-8"><h3>无法打开人工复核</h3>'
                f'<pre>{html.escape(str(exc))}</pre><a href="/">返回主界面</a>', 400)
    host = request.host.split(':', 1)[0]
    return redirect(f'http://{host}:7863/')


@app.get('/api/review/status')
def api_review_status():
    try:
        _, ctx = _context_for_video(request.args.get('video', ''))
        return jsonify(ok=True, status=ctx.get('status', 'pending_review'),
                       review_version=int(ctx.get('review_version') or 0),
                       main_video=ctx.get('main_video'),
                       annotations=(ctx.get('repaired_annotations') or
                                    ctx.get('candidate_annotations')),
                       frame_labels=(ctx.get('repaired_frame_labels') or
                                     ctx.get('candidate_frame_labels')))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400


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
