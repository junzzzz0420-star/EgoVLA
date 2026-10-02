from pathlib import Path


p = Path(
"ego_pipeline/preprocessing/das/fuse_multiview.py"
)

s = p.read_text()


old = """
        cal = json.loads(
            Path(glob.glob(f'{stereo_cal_dir}/{cam}_rect.mp4.json')[0]).read_text()
        )
"""


new = """
        cal_path = glob.glob(
            f'{stereo_cal_dir}/{cam}_rect.mp4.json'
        )[0]

        raw_cal = json.loads(
            Path(cal_path).read_text()
        )

        # 兼容 EgoCap / DAS 两种格式
        if 'calibration' in raw_cal:
            cal = raw_cal['calibration']
        else:
            cal = raw_cal

        if 'T_b_c_txyz_qxyzw' not in cal:
            cal['T_b_c_txyz_qxyzw'] = [
                0,
                0,
                0,
                0,
                0,
                0,
                1
            ]

        print(
            '[stereo calib]',
            cam,
            cal.keys()
        )
"""


if old not in s:
    print("target not found")
else:
    s=s.replace(old,new)
    p.write_text(s)

    print("patched fuse calibration")
