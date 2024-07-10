from flask import Flask, Response, render_template, request, jsonify
from picamera2 import Picamera2 
import libcamera
import cv2
import atexit
import numpy as np
import signal
import os

"""
PI 5 Version
This script displays the video live stream to browser with a button "save image".
When you click the button, the current frame will be saved to "/images"
visit: http://your_mbot_ip:5001
"""

class Camera:
    def __init__(self, camera_id, width, height):
        self.cap = Picamera2(camera_id)
        config = self.cap.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        config["transform"] = libcamera.Transform(hflip=0, vflip=1)
        self.cap.align_configuration(config)
        self.cap.configure(config)
        self.cap.start()
        self.image_count = 0
        self.frame = None
        self.running = True

    def get_frame(self):
        return self.cap.capture_array()

    def generate_frames(self):
        while self.running:
            frame = self.get_frame()
            # Encode the frame 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def cleanup(self):
        self.running = False
        self.cap.close()
        print("Camera resources released")

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    camera.cleanup()  # Clean up camera resources
    os._exit(0)  # Exit the program
    
app = Flask(__name__)

@app.route('/')
def video_page():
    return render_template('image_save_page.html')

@app.route('/video')
def video():
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save-image', methods=['POST'])
def save_image():
    # Extract the image save path from the POST request
    camera.image_count += 1
    save_path = "images/image"+str(camera.image_count)+".jpg"
    frame = camera.get_frame()
    if frame is not None:
        # Save the captured frame to the specified path
        cv2.imwrite(save_path, frame)
        return jsonify({"message": "Image saved successfully", "path": save_path}), 200
    else:
        return jsonify({"message": "Failed to capture image"}), 500

if __name__ == '__main__':
    # image width and height here should align with apriltag_streamer.py
    camera_id = 0
    image_width = 1280
    image_height = 720
    camera = Camera(camera_id, image_width, image_height)
    atexit.register(camera.cleanup)
    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler
    app.run(host='0.0.0.0', port=5001)