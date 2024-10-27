from flask import Flask, Response
from utils.utils import register_signal_handlers
from utils.camera_with_apriltag import CameraWithAprilTag
from utils.config import CAMERA_CONFIG, CONE_CONFIG
from ultralytics import YOLO
import cv2
import numpy as np

"""
url: http://your_mbot_ip:5001
"""

class CameraYOLO(CameraWithAprilTag):
    def __init__(self, camera_id, width, height, calibration_data, ncnn_model, frame_duration=None):
        super().__init__(camera_id, width, height, calibration_data, frame_duration)
        self.model = ncnn_model
        self.results = None
        self.cone_base_radius = CONE_CONFIG["cone_base_radius"]
        self.cone_height = CONE_CONFIG["cone_height"]

        self.cone_object_points = np.array([
            [self.cone_base_radius, 0, 0],  # Point at base circumference
            [-self.cone_base_radius, 0, 0], # Opposite point at base circumference
            [0, 0, 0],                 # Center of the base
            [0, 0, -self.cone_height]       # Apex of the cone
        ], dtype=np.float32)

    def process_frame(self, frame):
        # Increment frame count and check if it's time for detection
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            # Convert frame to grayscale and detect tags
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.detections = self.retry_detection(gray, retries=3)
            self.results = self.model(frame)

        

        if self.results:
            # frame = self.results[0].plot()
            # Iterate through detections and find cones
            for detection in self.results[0].boxes:
                class_id = int(detection.cls[0])  # Get class ID
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = detection.xyxy[0]

                # Calculate the coordinates for the front edge (base center of the bounding box)
                front_x = (x_min + x_max) / 2
                front_y = y_max  # Bottom of the bounding box

                # Calculate the height of the cone in the image
                image_cone_height = y_max - y_min

                # Calculate distance using triangle similarity
                if image_cone_height > 0:  # Avoid division by zero
                    focal_length = self.camera_matrix[1, 1]  # Approximate focal length from calibration data
                    z_distance = (focal_length * self.cone_height) / image_cone_height
                else:
                    z_distance = -1  # Error indicator

                # Calculate horizontal offset from the image center
                image_center_x = self.camera_matrix[0, 2]
                cone_center_x = (x_min + x_max) / 2
                delta_x = cone_center_x - image_center_x

                # Calculate the x-distance using the focal length
                fx = self.camera_matrix[0, 0]  # Focal length in x-direction
                x_distance = (z_distance * delta_x) / fx

                # Annotate frame with the distance information
                distance_text = f"X: {x_distance:.2f}mm, Z: {z_distance:.2f}mm"
                cv2.putText(frame, distance_text, (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # Annotate frame if detections are present
        if self.detections:
            for idx, detection in enumerate(self.detections):
                self.draw_tag_and_label(frame, detection, idx)

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
    calibration_data = np.load('cam_calibration_data.npz')
    # Load a YOLO11n PyTorch model
    # model = YOLO("example_model.pt")

    # Export the model to NCNN format
    # model.export(format="ncnn")

    # Load the exported NCNN model
    ncnn_model = YOLO("example_model_ncnn_model")

    camera = CameraYOLO(camera_id, image_width, image_height, calibration_data, ncnn_model, frame_duration)
    register_signal_handlers(camera.cleanup)

    app.run(host='0.0.0.0', port=5001)
