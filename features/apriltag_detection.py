#!/usr/bin/env python3
from flask import Flask, Response
import numpy as np
import time

from utils.utils import register_signal_handlers
from utils.config import CAMERA_CONFIG
from camera.camera import Camera
from camera.apriltag_detector import AprilTagDetector

"""
Features:
1. Displays the video live stream with apriltag detection to the browser.
2. Display the pose estimate values.

visit: http://your_mbot_ip:5001
"""

class AprilTagViewer(Camera):
    def __init__(self, camera_id, width, height, calibration_data, fps=None):
        super().__init__(camera_id, width, height, fps)
        self.tag_detector = AprilTagDetector(calibration_data)

    def process_frame(self, frame):
        self.tag_detector.detect_tags(frame)
        self.tag_detector.draw_tags(frame)

        return frame

app = Flask(__name__)
@app.route('/')
def video():
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # setup camera
    config = CAMERA_CONFIG
    camera_id = config["camera_id"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    fps = config["fps"]

    calibration_data = np.load('cam_calibration_data.npz')

    camera = AprilTagViewer(camera_id, image_width, image_height,
                                      calibration_data, fps)
    register_signal_handlers(camera.cleanup)
    app.run(host='0.0.0.0', port=5001)
