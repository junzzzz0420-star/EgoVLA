import os
import cv2
import json
import argparse
import numpy as np

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


LEFT_TOPIC = "/egocap/camera/left/image_raw"
RIGHT_TOPIC = "/egocap/camera/right/image_raw"
IMU_TOPIC = "/egocap/imu/data_raw"

typestore = get_typestore(Stores.ROS2_HUMBLE)


def decode_image(msg):
    img = np.frombuffer(
        msg.data,
        dtype=np.uint8
    )

    img = img.reshape(
        msg.height,
        msg.width,
        2
    )

    return cv2.cvtColor(
        img,
        cv2.COLOR_YUV2BGR_YUY2
    )


def export(input_path, output):

    os.makedirs(output, exist_ok=True)

    writers = {}

    timestamps = {
        "left": [],
        "right": []
    }

    imu_t = []
    imu_g = []
    imu_a = []

    frame_count = {
        "left": 0,
        "right": 0
    }


    with open(input_path, "rb") as f:

        reader = make_reader(f)

        for schema, channel, raw in reader.iter_messages():

            topic = channel.topic

            if topic not in [
                LEFT_TOPIC,
                RIGHT_TOPIC,
                IMU_TOPIC
            ]:
                continue


            if topic in [
                LEFT_TOPIC,
                RIGHT_TOPIC
            ]:

                typestore.register(
                    get_types_from_msg(
                        schema.data.decode(),
                        schema.name
                    )
                )

                msg = typestore.deserialize_cdr(
                    raw.data,
                    schema.name
                )

                img = decode_image(msg)

                side = (
                    "left"
                    if "left" in topic
                    else "right"
                )


                if side not in writers:

                    h, w = img.shape[:2]

                    writers[side] = cv2.VideoWriter(
                        os.path.join(
                            output,
                            f"{side}.mp4"
                        ),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30,
                        (w, h)
                    )


                writers[side].write(img)

                timestamps[side].append(
                    raw.log_time
                )

                frame_count[side] += 1


            elif topic == IMU_TOPIC:

                typestore.register(
                    get_types_from_msg(
                        schema.data.decode(),
                        schema.name
                    )
                )

                msg = typestore.deserialize_cdr(
                    raw.data,
                    schema.name
                )


                imu_t.append(raw.log_time)

                imu_g.append([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])

                imu_a.append([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])


    for w in writers.values():
        w.release()


    np.save(
        os.path.join(output, "left_timestamps.npy"),
        np.asarray(timestamps["left"])
    )

    np.save(
        os.path.join(output, "right_timestamps.npy"),
        np.asarray(timestamps["right"])
    )


    np.savez_compressed(
        os.path.join(output, "imu.npz"),
        t_log_ns=np.asarray(
            imu_t,
            dtype=np.int64
        ),
        gyro_rad_s=np.asarray(
            imu_g,
            dtype=np.float64
        ).reshape(-1,3),
        accel_g=np.asarray(
            imu_a,
            dtype=np.float64
        ).reshape(-1,3)
    )


    with open(
        os.path.join(output, "calib.json"),
        "w"
    ) as f:

        json.dump(
            {
                "source": "EgoCap",
                "width": 640,
                "height": 400,
                "encoding": "yuv422_yuy2",
                "fps": 30,
                "frames": frame_count,
                "imu_samples": len(imu_t)
            },
            f,
            indent=2
        )


    print("frames:", frame_count)
    print("imu samples:", len(imu_t))
    print("done:", output)



if __name__ == "__main__":

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

    export(
        args.input,
        args.output
    )
