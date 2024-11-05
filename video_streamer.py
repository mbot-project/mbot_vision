from flask import Flask, Response
from utils.utils import register_signal_handlers
from camera.camera import Camera
from utils.config import CAMERA_CONFIG

"""
This script only displays the video live stream to browser.
This is a simple program to check if the camera work.
url: http://your_mbot_ip:5001
"""

# setup camera
config = CAMERA_CONFIG
camera_id = config["camera_id"]
image_width = config["image_width"]
image_height = config["image_height"]
fps = config["fps"]

camera = Camera(camera_id, image_width, image_height, fps)
register_signal_handlers(camera.cleanup)

# create flask app
app = Flask(__name__)
@app.route('/')
def video():
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
