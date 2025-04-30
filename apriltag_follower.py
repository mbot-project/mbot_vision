#!/usr/bin/env python3
from flask import Flask, Response
import numpy as np
import threading

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
        self.latest_frame = None

    def process_frame(self, frame):
        self.tag_detector.draw_tags(frame)
        return frame

app = Flask(__name__)
@app.route('/')
def video():
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Processing thread function
def detection_thread():
    while camera.running:
        frame = camera.capture_frame()
        camera.tag_detector.detect_tags(frame)
        camera.tag_detector.follow_apriltag()
        camera.latest_frame = frame  # Store processed frame
    
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
    
    def follower_cleanup():
        camera.tag_detector.stop_robot()
        camera.cleanup()
    
    register_signal_handlers(follower_cleanup)
    
    # Start detection in separate thread
    processing_thread = threading.Thread(target=detection_thread)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Keep main thread alive
    while True:
        threading.Event().wait(1)
