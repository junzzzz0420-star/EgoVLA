from pathlib import Path

p = Path("ego_pipeline/preprocessing/das/run_multiview.py")

s = p.read_text(encoding="utf-8")


old = """        rect = export / f'{cam}_rect.mp4'
        if not rect.exists():
            run([PY, HERE / 'undistort_ds.py',
                 '--calib', export / f'calib_{cam}.json',
                 '--video', export / f'{cam}.mp4',
                 '--out', rect, '--hfov', args.hfov])
"""

new = """        rect = export / f'{cam}_rect.mp4'

        # EgoCap 已经输出 pinhole 图像，无 double-sphere 畸变
        # 直接复用原视频
        if args.input_export:
            rect = export / f'{cam}.mp4'
        elif not rect.exists():
            run([PY, HERE / 'undistort_ds.py',
                 '--calib', export / f'calib_{cam}.json',
                 '--video', export / f'{cam}.mp4',
                 '--out', rect, '--hfov', args.hfov])
"""


if old not in s:
    print("target not found")
else:
    s=s.replace(old,new)
    p.write_text(s,encoding="utf-8")
    print("patched:",p)
