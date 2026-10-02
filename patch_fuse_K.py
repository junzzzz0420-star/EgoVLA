from pathlib import Path

p = Path("ego_pipeline/preprocessing/das/fuse_multiview.py")

s = p.read_text(encoding="utf-8")


old = """        K = np.array(j['K'], float)
"""


new = """        if 'K' in j:
            K = np.array(j['K'], float)
        elif 'calibration' in j and 'K' in j['calibration']:
            K = np.array(j['calibration']['K'], float)
        else:
            # EgoCap fallback:
            # 640x400 stereo camera approximate intrinsics
            w = j.get('width', 640)
            h = j.get('height', 400)

            fx = fy = 0.9 * max(w, h)
            cx = w / 2.0
            cy = h / 2.0

            K = np.array([
                [fx,0,cx],
                [0,fy,cy],
                [0,0,1]
            ], dtype=float)

            print(
                '[fuse] warning: missing K, use approximate intrinsics',
                K
            )
"""


if old not in s:
    print("not found")
else:
    s=s.replace(old,new)
    p.write_text(s,encoding="utf-8")
    print("patched")


