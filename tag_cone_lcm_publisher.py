#!/usr/bin/env python3
import numpy as np
import time
from ultralytics import YOLO
from utils.utils import register_signal_handlers
from utils.config import CAMERA_CONFIG
from camera.camera import Camera
from camera.cone_detector import ConeDetector
from camera.apriltag_detector import AprilTagDetector

if __name__ == '__main__':
    # setup camera
    config = CAMERA_CONFIG
    camera_id = config["camera_id"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    fps = config["fps"]

    calibration_data = np.load('cam_calibration_data.npz')

    # Load the exported NCNN model
    model = YOLO("utils/best_ncnn_model", task='detect')

    camera = Camera(camera_id, image_width, image_height, fps)
    register_signal_handlers(camera.cleanup)
    cone_detector = ConeDetector(model, calibration_data)
    tag_detector = AprilTagDetector(calibration_data)

    while camera.running:
        frame = camera.capture_frame()
        cone_detector.detect_cones(frame)
        tag_detector.detect_tags(frame)
        cone_detector.publish_cones()
        tag_detector.publish_apriltag()


