import cv2
import lcm
import numpy as np
from utils.config import CONE_CONFIG
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t
from mbot_lcm_msgs.mbot_cone_t import mbot_cone_t
from utils.yolov8n.yolov8n_memryx import YOLOv8n
from memryx import AsyncAccl

class ConeDetectorMemryx:
    """
    Cone detector implementation using Memryx chip for acceleration.
    Maintains compatibility with original ConeDetector interface.
    """
    def __init__(self, calibration_data):
        config = CONE_CONFIG
        self.cone_height = config["cone_height"]
        self.conf_thres = config["conf_thres"]
        self.camera_matrix = calibration_data['camera_matrix']
        self.detections = []
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

        # Initialize YOLOv8n model
        self.model = YOLOv8n()
        
        # Initialize Memryx accelerator
        self.accl = AsyncAccl(self.model.dfp_path)
        self.accl.set_postprocessing_model(self.model.post_model)

        # Set up callbacks
        def output_callback(*fmaps):
            raw_dets = self.model.postprocess(fmaps)
            self.detections = self.cone_pose_estimate(raw_dets)
            
        self.accl.connect_output(output_callback)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Cleanup Memryx resources"""
        if hasattr(self, 'accl'):
            try:
                # Delete the accelerator object to release resources
                del self.accl
            except Exception as e:
                print(f"Error during Memryx cleanup: {e}")
            self.accl = None  # Ensure the reference is cleared

    def detect_cones(self, frame):
        """Process frame and detect cones using Memryx chip"""
        try:
            # Preprocess frame
            input_tensor = self.model.preprocess(frame)
            
            # Set up input callback
            def input_callback():
                return input_tensor
            
            # Connect input callback and start processing
            self.accl.connect_input(input_callback)
            
        except Exception as e:
            print(f"Error during detection: {e}")
            self.cleanup()  # Cleanup on error
            raise

    def cone_pose_estimate(self, raw_dets):
        """Convert raw detections to cone poses with distance estimation"""
        detection_results = []
        
        # raw_dets is a 2D array, first dimension is number of detections
        # Each detection has format [x1, y1, x2, y2, confidence, ...]
        try:
            dets = raw_dets[0]  # Get first batch of detections
            for i in range(len(dets)):
                # Extract bounding box coordinates and confidence
                x_min = float(dets[i][0])
                y_min = float(dets[i][1])
                x_max = float(dets[i][2])
                y_max = float(dets[i][3])
                confidence = float(dets[i][4])

                if confidence < float(self.conf_thres):
                    continue

                x_center = (x_min + x_max) / 2
                
                # Calculate distance using triangle similarity
                image_cone_height = y_max - y_min
                if image_cone_height > 0:
                    focal_length = self.camera_matrix[1, 1]
                    z_distance = (focal_length * self.cone_height) / image_cone_height
                    
                    # Calculate horizontal offset
                    image_center_x = self.camera_matrix[0, 2]
                    delta_x = x_center - image_center_x
                    fx = self.camera_matrix[0, 0]
                    x_distance = (z_distance * delta_x) / fx
                    
                    detection_results.append({
                        "class_name": "cone",
                        "confidence": confidence,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "x_distance": x_distance,
                        "z_distance": z_distance
                    })
        except Exception as e:
            print(f"Error processing detections: {e}")
            print(f"Raw detections shape: {raw_dets.shape if hasattr(raw_dets, 'shape') else 'unknown'}")

        return detection_results

    def draw_cone_detect(self, frame):
        """Draw detection results on frame"""
        for detection in self.detections:
            if detection["confidence"] < self.conf_thres:
                continue

            x_min = int(detection["x_min"])
            y_min = int(detection["y_min"])
            x_max = int(detection["x_max"])
            y_max = int(detection["y_max"])
            x_distance = detection["x_distance"]
            z_distance = detection["z_distance"]
            confidence = detection["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Add text annotations
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)
            y_offset = 15

            texts = [
                f"Cone: {confidence:.2f}",
                f"X: {x_distance:.2f}mm",
                f"Z: {z_distance:.2f}mm"
            ]

            for i, text in enumerate(texts):
                y_pos = y_min - y_offset - (20 * i)
                cv2.putText(frame, text, (x_min, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)
                cv2.putText(frame, text, (x_min, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        return frame

    def publish_cones(self):
        """Publish cone detections to LCM"""
        msg = mbot_cone_array_t()
        msg.array_size = 0
        msg.detections = []
        
        for detection in self.detections:
            if detection["confidence"] > self.conf_thres:
                cone = mbot_cone_t()
                cone.name = detection["class_name"]
                cone.pose.x = detection["x_distance"]
                cone.pose.z = detection["z_distance"]
                msg.detections.append(cone)
                msg.array_size += 1

        self.lcm.publish("MBOT_CONE_ARRAY", msg.encode()) 