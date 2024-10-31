from flask import Flask, Response
from utils.utils import register_signal_handlers
from camera.camera import Camera
from utils.config import CAMERA_CONFIG
from ultralytics import YOLO
import cv2
"""
To test if your YOLO model works

url: http://your_mbot_ip:5001
"""

# Load a YOLO11n PyTorch model
model = YOLO("../utils/cone_detection_model.pt")

# Export the model to NCNN format
model.export(format="ncnn")

# Load the exported NCNN model
ncnn_model = YOLO("../utils/cone_detection_model_ncnn_model")

class CameraYOLO(Camera):
    def __init__(self, camera_id, width, height, ncnn_model, fps=None):
        super().__init__(camera_id, width, height, fps)
        self.model = ncnn_model
        self.skip_frames = 5
        self.frame_count = 0
        self.results = None

    def process_frame(self, frame):
        # Increment frame count and check if it's time for detection
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            self.results = self.model(frame)

        if not self.results:
            return frame

        frame = self.results[0].plot()
        return frame

# create flask app
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

    camera = CameraYOLO(camera_id, image_width, image_height, ncnn_model, fps)
    register_signal_handlers(camera.cleanup)

    app.run(host='0.0.0.0', port=5001)
