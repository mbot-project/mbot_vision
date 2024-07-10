import lcm
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t

"""
This scripts subscribe to the MBOT_CONE_ARRAY
We use this program to check if the cone publisher work as expected
This is adapted from apriltag lcm publisher and works with raspberry pi 5
"""

def cone_callback(channel, data):
    msg = mbot_cone_array_t.decode(data)
    pos_text = ""
    if msg.array_size == 0:
        print("No Detection")
    else:
        for detection in msg.detections:
            color = detection.color
            range = detection.range
            heading = detection.heading
            pos_text += f"Color {color}: R={range:.2f}, H={heading:.2f}, "
        
        print(pos_text)


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
subscription = lc.subscribe("MBOT_CONE_ARRAY", cone_callback)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass
