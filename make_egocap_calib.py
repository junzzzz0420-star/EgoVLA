from pathlib import Path
import json


root = Path("examples/egocap_das_export/bag_0")


src = root / "calib_camera0.json"

data = json.loads(src.read_text())


# 找真正的外参字段
if "T_b_c_txyz_qxyzw" in data:
    T = data["T_b_c_txyz_qxyzw"]

elif "calibration" in data and "T_b_c_txyz_qxyzw" in data["calibration"]:
    T = data["calibration"]["T_b_c_txyz_qxyzw"]

else:
    print(data.keys())
    raise RuntimeError(
        "cannot find T_b_c_txyz_qxyzw"
    )


out = {
    "source": "EgoCap",

    "calibration": {
        "T_b_c_txyz_qxyzw": T
    },

    "camera0":{
        "file":"camera0.mp4"
    },

    "camera1":{
        "file":"camera1.mp4"
    },

    "imu":{
        "file":"imu.npz"
    }
}


(root/"calib.json").write_text(
    json.dumps(out,indent=2)
)


print("fixed:")
print(out["calibration"])
