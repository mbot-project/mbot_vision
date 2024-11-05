import lcm
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t
from mbot_lcm_msgs.mbot_apriltag_array_t import mbot_apriltag_array_t

"""
This scripts subscribe to the MBOT_CONE_ARRAY and  MBOT_APRILTAG_ARRAY
"""

def cone_callback(channel, data):
    msg = mbot_cone_array_t.decode(data)
    if msg.array_size == 0:
        print("No Cone Deteced")
    else:
        for detection in msg.detections:
            name = detection.name
            x = detection.pose.x
            z = detection.pose.z
            pos_text = f"{name}: x={x:.2f}, z={z:.2f}"
            print(pos_text)

def apriltag_callback(channel, data):
    msg = mbot_apriltag_array_t.decode(data)
    if msg.array_size == 0:
        print("No Tag Detected")
    else:
        for detection in msg.detections:
            tag_id = detection.tag_id
            [roll, pitch, yaw] = detection.pose.angles_rpy
            [qx, qy, qz, qw] = detection.pose.angles_quat
            pos_text = f"Tag ID {tag_id}: x={detection.pose.x:.2f}, y={detection.pose.y:.2f}, z={detection.pose.z:.2f} "
            euler_text = f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}"
            print(pos_text+euler_text)


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
lc.subscribe("MBOT_CONE_ARRAY", cone_callback)
lc.subscribe("MBOT_APRILTAG_ARRAY", apriltag_callback)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass