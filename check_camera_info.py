from mcap.reader import make_reader
from rosbags.typesys import Stores,get_typestore,get_types_from_msg


mcap="/mnt/d/work/ego/LYZ/egocap_20261001_060656/bag/bag_0.mcap"


typestore=get_typestore(
    Stores.ROS2_HUMBLE
)


topics=[
"/egocap/camera/left/camera_info",
"/egocap/camera/right/camera_info"
]


with open(mcap,"rb") as f:

    reader=make_reader(f)

    for schema,channel,msg in reader.iter_messages():

        if channel.topic in topics:

            print("\nTOPIC:",channel.topic)

            typestore.register(
                get_types_from_msg(
                    schema.data.decode(),
                    schema.name
                )
            )

            data=typestore.deserialize_cdr(
                msg.data,
                schema.name
            )


            print(data)

            break
