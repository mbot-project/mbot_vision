import cv2
from apriltag import apriltag
from utils.config import TAG_CONFIG
from utils.utils import rotation_matrix_to_euler_angles, rotation_matrix_to_quaternion
import numpy as np
import time
import lcm
from mbot_lcm_msgs.mbot_apriltag_array_t import mbot_apriltag_array_t
from mbot_lcm_msgs.mbot_apriltag_t import mbot_apriltag_t
from mbot_lcm_msgs.twist2D_t import twist2D_t
import math

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

    def follow_apriltag(self):
        """
        Follow the closest AprilTag detected by the camera.
        When no tag is detected, the robot will stop.
        """
        if self.detections:
            closest_tag = None
            min_z = float('inf')
            
            # Find the closest tag
            for detection in self.detections:
                x, y, z, roll, pitch, yaw, quaternion = self.decode_detection(detection)
                if z < min_z:
                    min_z = z
                    closest_tag = (x, z)
            

            x, z = closest_tag
            self.publish_velocity_command(x, z)
        else:
            # No detections - explicitly send a stop command
            self.stop_robot()
    
    def stop_robot(self):
        """
        Send a command to stop the robot completely.
        """
        command = twist2D_t()
        command.vx = 0.0
        command.wz = 0.0
        self.lcm.publish("MBOT_VEL_CMD", command.encode())

    def publish_velocity_command(self, x, z):
        """
        Publish a velocity command based on the x and z offset of the detected tag.
        """
        # Safety check - if z is very small or zero, stop the robot
        if z < 1.0:  # Unrealistically close or invalid measurement
            self.stop_robot()
            return
            
        # Constants
        k_p = 0.002  # Reduced proportional gain for linear velocity
        k_p_reverse = 0.002  # Higher gain for backing up (more responsive)
        k_theta = 2.0  # Increased angular gain for faster turning
        k_center = 0.8  # Additional gain for centering
        z_target = 250  # Target distance (millimeters)
        backup_threshold = 200  # Distance threshold to start backing up
        max_linear_speed = 0.15  # Maximum linear speed
        max_reverse_speed = 0.12  # Maximum reverse speed
        min_linear_speed = 0.03  # Minimum forward speed when following
        max_angular_speed = 1.2  # Increased maximum angular speed
        
        # Calculate angle to target with stronger centering bias
        theta = math.atan2(x, z)  # Angle to target
        
        # Add extra centering component - this increases turning when off-center
        centering_factor = k_center * (x / (z + 100))  # Normalized lateral offset
        wz = -k_theta * (theta + centering_factor)  # Angular velocity with centering bias
        
        # Limit angular velocity
        wz = max(min(wz, max_angular_speed), -max_angular_speed)
        
        # Calculate linear velocity based on distance to target
        if z > z_target:
            # Go forward when target is too far
            error = z - z_target
            vx = k_p * error
            
            # Reduce speed when turning sharply or when far off-center
            turn_factor = 1.0 - (min(abs(wz), max_angular_speed) / max_angular_speed) * 0.7
            center_factor = 1.0 - min(abs(x) / 100, 0.5)  # Reduce speed more when off-center
            vx = vx * turn_factor * center_factor
            
            # Apply limits
            vx = max(min(vx, max_linear_speed), min_linear_speed)
        elif z < backup_threshold:
            # Back up when too close
            error = backup_threshold - z
            vx = -k_p_reverse * error  # Negative velocity for backing up
            
            # Apply reverse speed limit
            vx = max(-max_reverse_speed, vx)
        else:
            # In the "dead zone" between backup_threshold and z_target
            if z > backup_threshold * 0.7:
                vx = min_linear_speed * 0.5  # Crawl forward slowly
            else:
                vx = 0  # Stop if in the middle of the "dead zone"

        # Create and publish the velocity command
        command = twist2D_t()
        command.vx = vx
        command.wz = wz
        self.lcm.publish("MBOT_VEL_CMD", command.encode())