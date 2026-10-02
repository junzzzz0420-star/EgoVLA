from mcap.reader import make_reader


mcap = "/mnt/d/work/ego/LYZ/egocap_20261001_060656/bag/bag_0.mcap"


topics=set()


with open(mcap,"rb") as f:
    reader=make_reader(f)

    for schema, channel, msg in reader.iter_messages():
        topics.add(channel.topic)


for t in sorted(topics):
    print(t)
