#!/usr/bin/env python3
from flask import Flask, Response
import numpy as np
from ultralytics import YOLO

from utils.utils import register_signal_handlers
from utils.config import CAMERA_CONFIG
from camera.camera import Camera
from camera.cone_detector import ConeDetector

"""
Features:
1. Displays the video live stream with cone detection to the browser.
2. Display the pose estimate values.

visit: http://your_mbot_ip:5001
"""

class ConeViewer(Camera):
    def __init__(self, camera_id, width, height, model, calibration_data, fps=None):
        super().__init__(camera_id, width, height, fps)
        self.cone_detector = ConeDetector(model, calibration_data)

    def process_frame(self, frame):
        self.cone_detector.detect_cones(frame)
        self.cone_detector.draw_cone_detect(frame)

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

    # Load a YOLO PyTorch model
    # model = YOLO("utils/cone_detection_model.pt")

    # Export the model to NCNN format
    # model.export(format="ncnn")

    # Load the exported NCNN model
    ncnn_model = YOLO("utils/cone_detection_model_ncnn_model")

    camera = ConeViewer(camera_id, image_width, image_height,
                                    ncnn_model, calibration_data, fps)
    register_signal_handlers(camera.cleanup)
    app.run(host='0.0.0.0', port=5001)
