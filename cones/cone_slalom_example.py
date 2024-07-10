#!/usr/bin/python3

import time
import lcm
import sys
import operator
import numpy as np
from mbot_lcm_msgs.twist2D_t import twist2D_t
from mbot_lcm_msgs.path2D_t import path2D_t
from mbot_lcm_msgs.pose2D_t import pose2D_t
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t

# Moves forward for 5 seconds, then stops
# If nothing happens, try running "sudo systemctl stop mbot-motion-controller.service", then running this file again

class Cone():
    def __init__(self, color, range, heading):
        self.color = color
        self.range = range
        self.heading = heading

class Slalom():
    def __init__(self):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        self.path = path2D_t()
        self.nextPose = pose2D_t()
        self.cones = []
        self.currentCone = None
        self.nextCone = None
        cone_subscription = self.lc.subscribe("MBOT_CONE_ARRAY", self.find_cones)

        reset = pose2D_t()
        reset.utime = int(time.time_ns()/1000)
        reset.x = 0.0
        reset.y = 0.0
        reset.theta = 0.0
        self.path.path.append(reset)
        self.path.path_length += 1
        self.lc.publish("MBOT_ODOMETRY_RESET", reset.encode())

        while (len(self.cones) == 0):
            self.lc.handle()
            print(len(self.cones))

        self.cones.sort(key=operator.attrgetter('range'))

        self.drive_cones()
        
    def find_cones(self, channel, data):
        msg = mbot_cone_array_t.decode(data)
        if msg.array_size == 0:
            print("No Detection")
        else:
            for detection in msg.detections:
                cone = Cone(detection.color, detection.range, detection.heading)
                self.cones.append(cone)

    def drive_cones(self):
        # for cone in self.cones:
        #     print(str(cone.color) + " " + str(cone.range) + " " + str(cone.heading))

        y_offset = 2.0 * self.cones[0].range * np.sin(self.cones[0].heading)/-100
        prev_y = 0.0
        next_x = None

        for i in range(1, len(self.cones)):
            next_x = (self.cones[i-1].range + self.cones[i].range)/200
            next_pose = pose2D_t()
            next_pose.x = next_x
            next_pose.y = prev_y
            next_pose.theta = 0.0
            self.path.path.append(next_pose)
            self.path.path_length += 1
            if (self.cones[i].color != self.cones[i-1].color):
                if (self.cones[i].color == "red_cone"): #cones not same color, next one is red (go right)
                    next_pose_red = pose2D_t()
                    next_pose_red.x = next_x
                    next_pose_red.y = 0.0
                    next_pose_red.theta = 0.0
                    self.path.path.append(next_pose_red)
                    self.path.path_length += 1
                    prev_y = 0.0
                else: #cones not same color, next one is green (go left)
                    next_pose_green = pose2D_t()
                    next_pose_green.x = next_x
                    next_pose_green.y = y_offset
                    next_pose_green.theta = 0.0
                    self.path.path.append(next_pose_green)
                    self.path.path_length += 1
                    prev_y = y_offset

        final_pose = pose2D_t()
        final_pose.x = next_x + 0.5
        final_pose.y = prev_y
        final_pose.theta = 0.0
        self.path.path.append(final_pose)
        self.path.path_length += 1

        self.path.utime = int(time.time_ns()/1000)
        self.lc.publish("CONTROLLER_PATH", self.path.encode())


my_slalom = Slalom()

# Edit these variables
# fwd_vel = 0.4
# turn_vel = 0.0
# move_time = 5

# command = twist2D_t() # A twist2D_t command encodes forward and rotational speeds of the bot
# command.vx = fwd_vel
# command.wz = turn_vel

# lc.publish("MBOT_VEL_CMD",command.encode())
# time.sleep(move_time)

# command.vx = 0
# command.wz = 0
# lc.publish("MBOT_VEL_CMD",command.encode())