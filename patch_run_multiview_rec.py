from pathlib import Path

p = Path("ego_pipeline/preprocessing/das/run_multiview.py")

s = p.read_text(encoding="utf-8")

old = """    rec = Path(args.mcap).name.rsplit('.', 1)[0]           # 完整 DAS-Ego_..._hash
"""

new = """    if args.input_export:
        rec = Path(args.input_export).name
    else:
        rec = Path(args.mcap).name.rsplit('.', 1)[0]
"""

if old not in s:
    print("target not found")
else:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("patched:", p)
