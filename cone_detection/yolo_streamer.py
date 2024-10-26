from flask import Flask, Response
from utils.utils import register_signal_handlers
from utils.camera import Camera
from utils.config import CAMERA_CONFIG
from ultralytics import YOLO
import cv2
"""
This script only displays the video live stream to browser.
This is a simple program to headlessly check if the camera work.
url: http://your_mbot_ip:5001
"""

class CameraYOLO(Camera):
    def __init__(self, camera_id, width, height, ncnn_model, frame_duration=None):
        super().__init__(camera_id, width, height, frame_duration)
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
    camera_id = CAMERA_CONFIG["camera_id"]
    image_width = CAMERA_CONFIG["image_width"]
    image_height = CAMERA_CONFIG["image_height"]
    fps = 20
    frame_duration = int((1./fps) * 1e6)

    # Load a YOLO11n PyTorch model
    model = YOLO("example_model.pt")

    # Export the model to NCNN format
    model.export(format="ncnn")

    # Load the exported NCNN model
    ncnn_model = YOLO("example_model_ncnn_model")

    camera = CameraYOLO(camera_id, image_width, image_height, ncnn_model, frame_duration)
    register_signal_handlers(camera.cleanup)

    app.run(host='0.0.0.0', port=5001)
