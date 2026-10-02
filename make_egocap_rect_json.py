from pathlib import Path
import json


root = Path("examples/egocap_das_export/bag_0")


for cam in ["camera0","camera1"]:

    w = 640
    h = 400


    fx = fy = 0.9 * max(w,h)
    cx = w/2
    cy = h/2


    K = [
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
    ]


    out = {

        "width": w,
        "height": h,

        "K": K,


        "calibration":{

            "K":K,
            "width":w,
            "height":h

        },


        # DAS格式要求:
        # translation xyz + quaternion xyzw
        # EgoCap没有外参，先单位变换

        "T_b_c_txyz_qxyzw":[
            0,
            0,
            0,
            0,
            0,
            0,
            1
        ]

    }


    dst = root / f"{cam}_rect.mp4.json"

    with open(dst,"w") as f:
        json.dump(out,f,indent=2)


    print("updated",dst)
