from pathlib import Path

p = Path("ego_pipeline/preprocessing/das/run_multiview.py")

s = p.read_text(encoding="utf-8")


# 1. 添加 input-export 参数
old = """    ap.add_argument('--mcap', required=True, help='原始 mcap 路径（录制标识取自文件名）')
"""

new = """    ap.add_argument('--mcap', required=False, default=None,
                    help='原始 mcap 路径（DAS格式）')
    ap.add_argument('--input-export', default=None,
                    help='已有导出目录(camera0.mp4,camera1.mp4,imu.npz)')
"""

if old in s:
    s = s.replace(old, new)
else:
    print("warning: mcap arg not found")


# 2. 修改 export 路径逻辑
old = """    export = Path(args.export_root) / rec
"""

new = """    if args.input_export:
        export = Path(args.input_export)
    else:
        export = Path(args.export_root) / rec
"""

if old in s:
    s = s.replace(old, new)
else:
    print("warning: export line not found")


# 3. 修改 need 判断
old = """    need = [c for c in cams
            if not (export / f'{c}.mp4').exists()
            or not (export / f'calib_{c}.json').exists()]
"""

new = """    need = []

    if not args.input_export:
        need = [c for c in cams
                if not (export / f'{c}.mp4').exists()
                or not (export / f'calib_{c}.json').exists()]
"""

if old in s:
    s = s.replace(old, new)
else:
    print("warning: need block not found")


# 4. mcap export 保持原样，但只有 need 时执行
p.write_text(s, encoding="utf-8")

print("patch finished:", p)
