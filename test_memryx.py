from flask import Flask, Response
from utils.utils import register_signal_handlers
from camera.camera import Camera
from memryx import AsyncAccl
from utils.config import CAMERA_CONFIG, CONE_CONFIG
from utils.metrics_logger import MetricsLogger
import numpy as np
import cv2
import signal
import threading
import sys

# Define class names for your cone classes
CLASS_NAMES = [
    'blue_cone',
    'green_cone',
    'pink_cone',
    'red_cone',
    'yellow_cone'
]

# Global variables for cleanup
camera = None
accl = None
metrics_logger = None

def signal_handler(signum, frame):
    print("\nCleaning up...")
    if camera:
        camera.cleanup()
    sys.exit(0)

class MemryxCamera(Camera):
    def __init__(self, camera_id, width, height, model, calibration_data, fps=None, confidence_thres=0.8, iou_thres=0.6):
        super().__init__(camera_id, width, height, fps)
        self.model = model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_width = 640  # Set input width
        self.input_height = 640  # Set input height
        self.detections = []
        self.camera_matrix = calibration_data['camera_matrix']
        config = CONE_CONFIG
        self.cone_height = config["cone_height"]

    def capture_and_preprocess(self):
        frame = self.capture_frame()
        # Detailed preprocessing to match yolov8.py
        original_img = frame
        img_height, img_width, _ = original_img.shape
        self.length = max(img_height, img_width)  # Store length as an instance attribute
        image = np.zeros((self.length, self.length, 3), np.uint8)
        image[0:img_height, 0:img_width] = original_img

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)
        blob = blob.squeeze(0)  # Removes the batch dimension -> (3, 640, 640)
        blob = blob.transpose(1, 2, 0)  # Change to (640, 640, 3)
        blob = np.expand_dims(blob, axis=2)  # Add new axis at index 2 -> (640, 640, 1, 3)
        return blob

    def postprocess_and_print(self, *mxa_output):
        """
        Post-processes the model output using the post-processing model.
        The post-processing model handles the output transformation.
        """
        # Clear previous detections at the start
        self.detections = []
        
        # Get the processed output from the post-processing model
        outputs = np.transpose(np.squeeze(mxa_output[0]))

        # Extract boxes and scores
        boxes = outputs[:, :4]  # x_center, y_center, width, height
        class_scores = outputs[:, 4:]  # 5 cone classes
        # Get confidence scores and class IDs
        confidence = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidence >= self.confidence_thres
        if not np.any(mask):
            return []
            
        # Apply confidence mask
        boxes = boxes[mask]
        confidence = confidence[mask]
        class_ids = class_ids[mask]
        
        # Convert normalized coordinates to pixel coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height
        
        # Convert box coordinates to corner format
        boxes_corner = np.zeros_like(boxes)
        boxes_corner[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * x_factor  # left
        boxes_corner[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * y_factor  # top
        boxes_corner[:, 2] = boxes[:, 2] * x_factor  # width
        boxes_corner[:, 3] = boxes[:, 3] * y_factor  # height
        
        # Create detection dictionaries
        detections = []
        for i in range(len(boxes_corner)):
            x_min, y_min, w, h = boxes_corner[i].astype(int).tolist()
            x_max = x_min + w
            y_max = y_min + h
            x_center = x_min + w / 2
            
            # Calculate distance using triangle similarity
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
            
            detections.append({
                'bbox': [x_min, y_min, w, h],
                'class_id': int(class_ids[i]),
                'class': CLASS_NAMES[int(class_ids[i])],
                'score': float(confidence[i]),
                'x_distance': x_distance,
                'z_distance': z_distance
            })
        
        # Apply NMS
        if detections:
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, 
                scores_for_nms, 
                self.confidence_thres, 
                self.iou_thres
            )
            
            if len(indices) > 0:
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]
                self.detections = [detections[i] for i in indices]
            
        # Print final detections
        print("\nFinal Detections after NMS:")
        for det in self.detections:
            print(f"Class: {det['class']}, Score: {det['score']:.4f}, BBox: {det['bbox']}")
        return self.detections

    def process_frame(self, frame):
        """
        Draw detections on the frame.
        """
        for detection in self.detections:
            x1, y1, w, h = detection['bbox']
            class_name = detection['class']
            confidence = detection['score']
            x_distance = detection['x_distance']
            z_distance = detection['z_distance']

            # Draw bounding box
            color = (0, 255, 255)  # Yellow color for bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Annotate frame with class name, confidence, and distance information
            label_y_offset = 15
            text_color = (255, 255, 255)  # White text
            outline_color = (0, 0, 0)  # Black outline

            # Annotate class name
            cv2.putText(frame, class_name, (int(x1), int(y1) - label_y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, class_name, (int(x1), int(y1) - label_y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate confidence below the class name
            confidence_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, confidence_text, (int(x1), int(y1) - label_y_offset - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, confidence_text, (int(x1), int(y1) - label_y_offset - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate x and z distances below confidence
            distance_text_x = f"X: {x_distance:.2f}mm"
            distance_text_z = f"Z: {z_distance:.2f}mm"

            # Annotate x distance
            cv2.putText(frame, distance_text_x, (int(x1), int(y1) - label_y_offset - 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, distance_text_x, (int(x1), int(y1) - label_y_offset - 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate z distance
            cv2.putText(frame, distance_text_z, (int(x1), int(y1) - label_y_offset - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, distance_text_z, (int(x1), int(y1) - label_y_offset - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

        return frame

def record_metrics():
    """
    Record system metrics including FPS, CPU, and memory usage
    """
    if camera and metrics_logger:
        fps = camera.measured_fps
        metrics_logger.log_metrics(fps)

def main():
    try:
        # Connect the accelerator
        accl.connect_input(camera.capture_and_preprocess)
        accl.connect_output(camera.postprocess_and_print)
        accl.wait()
    except Exception as e:
        print(f"Error in processing thread: {e}")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def video():
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def flask_run():
    app.run(host='0.0.0.0', port=5001, use_reloader=False)

if __name__ == '__main__':
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup camera configuration
    config = CAMERA_CONFIG
    camera_id = config["camera_id"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    fps = config["fps"]
    
    # Load calibration data
    calibration_data = np.load('cam_calibration_data.npz')

    try:
        # Initialize MemryX accelerator
        accl = AsyncAccl(dfp='utils/yolov8n/yolo8n.dfp')
        
        # Set the post-processing model
        accl.set_postprocessing_model('utils/yolov8n/best_v8n_post.onnx')

        # Initialize camera with MemryxCamera
        camera = MemryxCamera(camera_id=camera_id, width=image_width, height=image_height, model=accl, calibration_data=calibration_data, fps=fps)
        
        # Start the main processing loop in a separate thread
        processing_thread = threading.Thread(target=main)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=flask_run)
        flask_thread.daemon = True
        flask_thread.start()
        
        # Initialize metrics logger
        metrics_logger = MetricsLogger()

        # Keep the main thread running and record metrics
        while True:
            record_metrics()
            threading.Event().wait(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        if camera:
            camera.cleanup()
        print("Cleanup complete") 