import rosbag
import sys

# python read_bag_ros.py path_to_file
bag = rosbag.Bag(sys.argv[1])

# loop through selescted topics in bag file
for topic, msg, t in bag.read_messages(topics=['/topic/example', '/topic2/example2']):
    if topic == '/topic/example':
        # data from this topic is stored in msg
        # process data from this topic here
        pass
    if topic == '/topic2/example2':
        # process data from second topic
        pass

bag.close()
