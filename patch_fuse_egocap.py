from pathlib import Path

p = Path("ego_pipeline/preprocessing/das/fuse_multiview.py")

s = p.read_text()

old = """    cals = {c: json.loads(Path(f'{stereo_cal_dir}/calib_{c}.json').read_text())['calibration']
            for c in (ref_cam, stereo_cam)}
"""

new = """    cals = {}

    for c in (ref_cam, stereo_cam):

        candidates = [
            f'{stereo_cal_dir}/calib_{c}.json',
            f'{stereo_cal_dir}/{c}_rect.mp4.json'
        ]

        cal_file = None
        for x in candidates:
            if Path(x).exists():
                cal_file = x
                break

        if cal_file is None:
            raise FileNotFoundError(
                f'cannot find calibration for {c}'
            )

        data = json.loads(Path(cal_file).read_text())

        # DAS format
        if 'calibration' in data:
            data = data['calibration']

        # EgoCap format
        if 'T_b_c_txyz_qxyzw' not in data:
            data['T_b_c_txyz_qxyzw'] = [
                0,0,0,
                0,0,0,1
            ]

        cals[c] = data

        print(
            '[fuse calib]',
            c,
            cal_file,
            data.keys()
        )
"""

if old not in s:
    print("target block not found")
else:
    s=s.replace(old,new)
    p.write_text(s)
    print("patched fuse_multiview.py")
