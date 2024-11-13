import cv2
import lcm
from utils.config import CONE_CONFIG
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t
from mbot_lcm_msgs.mbot_cone_t import mbot_cone_t

class ConeDetector:
    def __init__(self, model, calibration_data):
        self.model = model
        self.results = None
        config = CONE_CONFIG
        self.cone_height = config["cone_height"]
        self.skip_frames = config["skip_frames"]
        self.conf_thres = config["conf_thres"]
        self.camera_matrix = calibration_data['camera_matrix']
        self.class_names = self.model.names
        self.frame_count = 0
        self.detections = []
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    def detect_cones(self, frame):
        # Increment frame count and check if it's time for detection
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            self.results = self.model(frame)
            self.detections = self.cone_pose_estimate()

    def cone_pose_estimate(self):
        detection_results = []
        if self.results:
            # Iterate through detections and find cones
            for detection in self.results[0].boxes:
                class_id = int(detection.cls[0])  # Get class ID
                class_name = self.class_names.get(class_id, "Unknown")  # Get class name from the model
                confidence = detection.conf[0]  # Get confidence score

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

                # Store detection result
                detection_results.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "x_distance": x_distance,
                    "z_distance": z_distance
                })

        return detection_results

    def draw_cone_detect(self, frame):
        # Draw all detections on the frame
        for detection in self.detections:
            x_min = detection["x_min"]
            y_min = detection["y_min"]
            x_max = detection["x_max"]
            y_max = detection["y_max"]
            class_name = detection["class_name"]
            x_distance = detection["x_distance"]
            z_distance = detection["z_distance"]
            confidence = detection["confidence"]

            # Draw the bounding box with a visually friendly color
            color = (0, 255, 255)  # Bright yellow color for the bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

            # Annotate frame with the class name, confidence, and distance information
            label_y_offset = 15
            text_color = (255, 255, 255)  # White text
            outline_color = (0, 0, 0)  # Black outline

            # Annotate class name
            cv2.putText(frame, class_name, (int(x_min), int(y_min) - label_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, class_name, (int(x_min), int(y_min) - label_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate confidence below the class name
            confidence_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, confidence_text, (int(x_min), int(y_min) - label_y_offset - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, confidence_text, (int(x_min), int(y_min) - label_y_offset - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate x and z distances below confidence
            distance_text_x = f"X: {x_distance:.2f}mm"
            distance_text_z = f"Z: {z_distance:.2f}mm"

            # Annotate x distance
            cv2.putText(frame, distance_text_x, (int(x_min), int(y_min) - label_y_offset - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, distance_text_x, (int(x_min), int(y_min) - label_y_offset - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

            # Annotate z distance
            cv2.putText(frame, distance_text_z, (int(x_min), int(y_min) - label_y_offset - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)  # Outline
            cv2.putText(frame, distance_text_z, (int(x_min), int(y_min) - label_y_offset - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Text

        return frame

    def publish_cones(self):
        msg = mbot_cone_array_t()
        msg.array_size = 0
        msg.detections = []
        if len(self.detections) > 0:
            for detection in self.detections:
                # check if the detection was false positive
                if detection["confidence"] > self.conf_thres:
                    cone = mbot_cone_t()
                    cone.name = detection["class_name"]
                    cone.pose.x = detection["x_distance"]
                    cone.pose.z = detection["z_distance"]
                    msg.detections.append(cone)
                    msg.array_size += 1

        self.lcm.publish("MBOT_CONE_ARRAY", msg.encode())