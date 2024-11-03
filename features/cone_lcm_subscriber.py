import lcm
from mbot_vision_lcm import mbot_cone_array_t

"""
This scripts subscribe to the MBOT_CONE_ARRAY
We use this program to check if the cone publisher work as expected
"""

def cone_callback(channel, data):
    msg = mbot_cone_array_t.decode(data)
    if msg.array_size == 0:
        print("No Detection")
    else:
        for detection in msg.detections:
            name = detection.name
            x = detection.x
            z = detection.z
            pos_text = f"{name}: x={x:.2f}, z={z:.2f}"
            print(pos_text)


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
subscription = lc.subscribe("MBOT_CONE_ARRAY", cone_callback)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass