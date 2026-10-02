from pathlib import Path
import shutil
import json
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()


    src = Path(args.input)
    dst = Path(args.output)

    dst.mkdir(
        parents=True,
        exist_ok=True
    )


    # video
    shutil.copy(
        src/"left.mp4",
        dst/"camera0.mp4"
    )

    shutil.copy(
        src/"right.mp4",
        dst/"camera1.mp4"
    )


    # imu
    shutil.copy(
        src/"imu.npz",
        dst/"imu.npz"
    )


    # calib
    calib = json.load(
        open(src/"calib.json")
    )


    for cam in ["camera0","camera1"]:

        out = {
            "source":"EgoCap",
            "camera":cam,
            "calibration":calib
        }

        json.dump(
            out,
            open(
                dst/f"calib_{cam}.json",
                "w"
            ),
            indent=2
        )


    print("converted:")
    print(dst)


if __name__=="__main__":
    main()
