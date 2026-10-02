from pathlib import Path
import cv2
import numpy as np

root = Path("examples/egocap_das_export/bag_0")

fps = 30.0

for cam in ["camera0","camera1"]:

    video = root / f"{cam}.mp4"

    cap = cv2.VideoCapture(str(video))

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_real = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps_real > 1:
        fps = fps_real

    ts = np.arange(n,dtype=np.int64)

    # 纳秒时间轴
    ts = (ts * (1e9/fps)).astype(np.int64)

    out = root / f"{cam}_timestamps.npz"

    np.savez_compressed(
        out,
        log_time_ns=ts,
        header_stamp_ns=ts,
        frame_index=np.arange(n,dtype=np.int64)
    )

    print(
        "created",
        out,
        "frames",
        n,
        "fps",
        fps
    )
