import cv2
from apriltag import apriltag
from utils.config import TAG_CONFIG
from utils.utils import rotation_matrix_to_euler_angles, rotation_matrix_to_quaternion
import numpy as np
import time
import lcm
from mbot_lcm_msgs.mbot_apriltag_array_t import mbot_apriltag_array_t
from mbot_lcm_msgs.mbot_apriltag_t import mbot_apriltag_t

class AprilTagDetector:
    """
    Handles AprilTag detection in frames.
    """
    def __init__(self, calibration_data):
        # Initialize detector with configuration data
        config = TAG_CONFIG
        self.detector = apriltag(config["tag_family"], threads=1)
        self.skip_frames = config["skip_frames"]
        self.tag_size = config["tag_size"]
        self.small_tag_size = config["small_tag_size"]
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.frame_count = 0
        self.detections = dict()

        self.object_points = np.array([
            [-self.tag_size/2,  self.tag_size/2, 0],  # Top-left corner
            [ self.tag_size/2,  self.tag_size/2, 0],  # Top-right corner
            [ self.tag_size/2, -self.tag_size/2, 0],  # Bottom-right corner
            [-self.tag_size/2, -self.tag_size/2, 0],  # Bottom-left corner
        ], dtype=np.float32)
        self.small_object_points = np.array([
            [-self.small_tag_size/2,  self.small_tag_size/2, 0],  # Top-left corner
            [ self.small_tag_size/2,  self.small_tag_size/2, 0],  # Top-right corner
            [ self.small_tag_size/2, -self.small_tag_size/2, 0],  # Bottom-right corner
            [-self.small_tag_size/2, -self.small_tag_size/2, 0],  # Bottom-left corner
        ], dtype=np.float32)
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    def detect_tags(self, frame):
        """
        Detects AprilTags in the provided frame.
        :param frame: Frame from the camera.
        :return: List of detections.
        """
        # Increment frame count and check if it's time for detection
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            # Convert frame to grayscale and detect tags
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.detections = self.retry_detection(gray, retries=3)

    def retry_detection(self, gray_frame, retries=3):
        # prevent quit from one detection fail
        for attempt in range(retries):
            try:
                return self.detector.detect(gray_frame)
            except RuntimeError as e:
                if "Unable to create" in str(e) and attempt < retries - 1:
                    print(f"Detection failed, retrying... Attempt {attempt + 1}")
                    time.sleep(0.2)
                else:
                    raise
        return ()  # Return an empty tuple if detection ultimately fails

    def draw_tags(self, frame):
        """
        Draws annotations for detected AprilTags on frames.
        """
        for idx, detection in enumerate(self.detections):
            self.draw_tag_and_label(frame, detection, idx)

    def draw_tag_and_label(self, frame, detection, idx):
        # Drawing and labeling logic for detected tags
        x, y, z, roll, pitch, yaw, quaternion = self.decode_detection(detection)
        corners = np.array(detection['lb-rb-rt-lt'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Position text annotation
        pos_text = f"Tag ID {detection['id']}: x={x:.2f}, y={y:.2f}, z={z:.2f},"
        orientation_text = f" roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}"
        vertical_pos = 40 * (idx + 1)
        text = pos_text + orientation_text

        # Draw text with outline
        text_color = (255, 255, 255)  # White text
        outline_color = (0, 0, 0)      # Black outline

        # Draw the outline first (black with a slightly larger thickness)
        cv2.putText(frame, text, (10, vertical_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline_color, 3)

        # Draw the actual text on top (white with a smaller thickness)
        cv2.putText(frame, text, (10, vertical_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def decode_detection(self, detect):
        """
        Decodes the pose information from a detected tag.
        :param detection: AprilTag detection.
        :return: Position and orientation (x, y, z, roll, pitch, yaw, quaternion).
        """
        if detect['id'] < 10:  # Big tag
            image_points = np.array(detect['lb-rb-rt-lt'], dtype=np.float32)
            retval, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if detect['id'] >= 10:  # Small tag at center
            image_points = np.array(detect['lb-rb-rt-lt'], dtype=np.float32)
            retval, rvec, tvec = cv2.solvePnP(self.small_object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        # Convert rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)

        return tvec[0][0], tvec[1][0], tvec[2][0], roll, pitch, yaw, quaternion

    def publish_apriltag(self):
        """
        Publish the apriltag message
        """
        msg = mbot_apriltag_array_t()
        msg.array_size = len(self.detections)
        msg.detections = []
        if msg.array_size > 0:
            for detection in self.detections:
                x, y, z, roll, pitch, yaw, quaternion = self.decode_detection(detection)

                apriltag = mbot_apriltag_t()
                apriltag.tag_id = detection['id']
                apriltag.pose.x = x
                apriltag.pose.y = y
                apriltag.pose.z = z
                apriltag.pose.angles_rpy = [roll, pitch, yaw]
                apriltag.pose.angles_quat = quaternion
                msg.detections.append(apriltag)

        self.lcm.publish("MBOT_APRILTAG_ARRAY", msg.encode())
