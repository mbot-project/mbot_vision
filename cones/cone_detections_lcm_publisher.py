from flask import Flask, Response
from picamera2 import Picamera2, Preview
import pandas as pd
from ultralytics import YOLO
import cvzone
import libcamera
import cv2
import time
import atexit
import numpy as np
import lcm
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t
from mbot_lcm_msgs.mbot_cone_t import mbot_cone_t

"""
Adapted from Apriltag LCM Publisher
Works with Raspberry Pi 5
Publishes to "MBOT_CONE_ARRAY"
"""

class Camera:
    def __init__(self, camera_id, width, height):

        # Define camera parameters and setup
        self.cap = Picamera2(camera_id)
        self.w = width
        self.h = height
        config = self.cap.create_preview_configuration(main={"size": (self.w, self.h), "format": "RGB888"})
        config["transform"] = libcamera.Transform(hflip=1, vflip=1)
        self.cap.align_configuration(config)
        self.cap.configure(config)
        self.cap.start()

        # Define YOLO model to use (input your trained model .pt here)
        self.model = YOLO('example_model.pt')

        # Define class list for detections
        f = open("example_classes.txt", "r")
        data = f.read()
        self.class_list = data.split("\n")
        self.frame_count = 0

        # Load calibration data (must be calibrated, use "save_image_rpi5.py" with "camera_calibration.py")
        # Calibration data must be used in the same frame size as this script
        calibration_data = np.load('../cam_calibration_data.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.w, self.h), 1, (self.w, self.h))
        self.x, self.y, self.w, self.h = roi

        self.arducam_focal_length_mm = 3.04 # mm, from arducam spec sheet
        self.arducam_focal_length_pixels = self.new_camera_matrix[1,1]
        self.m_scaled = self.arducam_focal_length_pixels / self.arducam_focal_length_mm # pixels per mm conversion factor
        self.center_pixel_line = self.w / 2 # this is just keeping track of the centerline in pixels of u, v coords

        self.cone_height_real = 80 # mm, actual cone height

        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    def generate_frames(self):
        while True:
            self.frame_count += 1
            frame = self.cap.capture_array()
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
            frame = frame[self.y : self.y + self.h, self.x : self.x + self.w]

            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results = self.model.predict(frame)
                    a = results[0].boxes.data
                    px = pd.DataFrame(a).astype("float")
                    self.num_detections = len(px.index)
                    self.detections = px.iterrows()
                    break  # Success, exit the retry loop
                except RuntimeError as e:
                    if "Unable to create" in str(e) and attempt < max_retries - 1:
                        print(f"Detection failed due to thread creation issue, retrying... Attempt {attempt + 1}")
                        time.sleep(0.2)  # back off for a moment
                    else:
                        raise  # Re-raise the last exception if retries exhausted

            self.publish_cones()
        

    def publish_cones(self):
        msg = mbot_cone_array_t()

        msg.array_size = self.num_detections
        msg.detections = []

        if msg.array_size > 0:
            for index, row in self.detections:
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = self.class_list[d]

                # find height of cone in u, v coords
                height = np.abs(y2 - y1)

                # scale cone height
                cone_height_sensor = height / self.m_scaled

                # calculate cone distance from camera (range) using similar triangles approach
                distance_cm = int((self.cone_height_real * self.arducam_focal_length_mm / cone_height_sensor) / 10)

                # calculate cone heading relative to camera view
                dist_from_center = (x1 + x2) / 2
                heading_difference = round(np.arctan((dist_from_center - self.center_pixel_line) / self.arducam_focal_length_pixels), 4)
                
                # create cone detection and append detections array 
                cone = mbot_cone_t()
                cone.color = c
                cone.range = distance_cm
                cone.heading = heading_difference
                msg.detections.append(cone)

        # publish detections array
        self.lcm.publish("MBOT_CONE_ARRAY", msg.encode())
        
    def cleanup(self):
        print("Releasing camera resources")
        self.cap.stop()

if __name__ == '__main__':
    # image width and height here should align with save_image.py
    camera_id = 0
    image_width = 1640
    image_height = 922
    camera = Camera(camera_id, image_width, image_height) 
    atexit.register(camera.cleanup)
    camera.generate_frames()

