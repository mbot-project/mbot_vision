#!/usr/bin/env python3
import numpy as np
from ultralytics import YOLO
from utils.utils import register_signal_handlers
from utils.config import CAMERA_CONFIG
from camera.camera import Camera
from camera.cone_detector import ConeDetector


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

    camera = Camera(camera_id, image_width, image_height, fps)
    register_signal_handlers(camera.cleanup)
    cone_detector = ConeDetector(ncnn_model, calibration_data)

    while camera.running:
        frame = camera.capture_frame()
        cone_detector.detect_cones(frame)
        cone_detector.publish_cones()


