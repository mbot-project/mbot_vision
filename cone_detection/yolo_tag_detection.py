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
        # Extract class names from the YOLO model
        self.class_names = self.model.names

    def process_frame(self, frame):
        # Increment frame count and check if it's time for detection
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            # Convert frame to grayscale and detect tags
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.detections = self.retry_detection(gray, retries=3)
            self.results = self.model(frame)

        if self.results:
            # frame = self.results[0].plot() # draw boxes, extremely slow...
            # Iterate through detections and find cones
            for detection in self.results[0].boxes:
                class_id = int(detection.cls[0])  # Get class ID
                class_name = self.class_names.get(class_id, "Unknown")  # Get class name from the model

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = detection.xyxy[0]
                x_center, y_center, box_width, box_height = detection.xywh[0]

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
                delta_x = x_center - image_center_x

                # Calculate the x-distance using the focal length
                fx = self.camera_matrix[0, 0]  # Focal length in x-direction
                x_distance = (z_distance * delta_x) / fx

                # Draw the bounding box with a more visually friendly color
                color = (0, 255, 255)  # Bright yellow color for the bounding box
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                # Annotate frame with the class name and distance information
                label_y_offset = 15
                # Use white text with a black outline for better visibility
                text_color = (255, 255, 255)  # White text
                outline_color = (0, 0, 0)  # Black outline

                # Annotate class name
                cv2.putText(frame, class_name, (int(x_min), int(y_min) - label_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
                cv2.putText(frame, class_name, (int(x_min), int(y_min) - label_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

                # Annotate distance below the class name, stacked vertically
                distance_text_x = f"X: {x_distance:.2f}mm"
                distance_text_z = f"Z: {z_distance:.2f}mm"

                # Annotate x distance
                cv2.putText(frame, distance_text_x, (int(x_min), int(y_min) - label_y_offset - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
                cv2.putText(frame, distance_text_x, (int(x_min), int(y_min) - label_y_offset - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

                # Annotate z distance
                cv2.putText(frame, distance_text_z, (int(x_min), int(y_min) - label_y_offset - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
                cv2.putText(frame, distance_text_z, (int(x_min), int(y_min) - label_y_offset - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

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
