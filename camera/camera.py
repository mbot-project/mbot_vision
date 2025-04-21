from picamera2 import Picamera2
import libcamera
import logging
import cv2
import time

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbosity during development
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Camera:
    """
    Base Camera class that provides functionality to interface with the Picamera2 camera.
    This class handles initialization, configuration, frame capturing, and frame streaming,
    while providing a framework for subclasses to implement custom frame processing.
    """
    def __init__(self, camera_id, width, height, fps=None):
        """
        Initializes the camera with the given parameters.
        :param camera_id: The ID of the camera to use.
        :param width: The width of the camera frame.
        :param height: The height of the camera frame.
        :param fps: Optional frame per second.
        """
        logging.info("Initializing camera...")
        self.cap = Picamera2(camera_id)
        config = self.cap.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        if fps:
            frame_duration = int((1./fps) * 1e6)
            config["controls"] = {'FrameDurationLimits': (frame_duration, frame_duration)}
        config["transform"] = libcamera.Transform(hflip=1, vflip=1)
        self.cap.align_configuration(config)
        self.cap.configure(config)
        self.cap.start()
        self.running = True
        
        # FPS tracking variables
        self.measured_fps = 0
        self.frame_time = time.time()
        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        
        logging.info("Camera initialized.")

    def get_fps(self):
        """
        Returns the current measured FPS.
        :return: Current frames per second measurement.
        """
        return self.measured_fps

    def capture_frame(self):
        """
        Captures a single frame from the camera.
        :return: The captured frame as a numpy array.
        """
        return self.cap.capture_array()

    def generate_frames(self):
        """
        Generates frames for streaming purposes.
        :return: A generator yielding encoded frames.
        """
        while self.running:
            # Update FPS calculation
            current_time = time.time()
            self.frame_count += 1
            
            if current_time - self.frame_time >= self.fps_update_interval:
                self.measured_fps = self.frame_count / (current_time - self.frame_time)
                self.frame_count = 0
                self.frame_time = current_time
            
            # Capture the frame
            frame = self.capture_frame()

            # Process the frame (to be defined in subclasses)
            frame = self.process_frame(frame)

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def process_frame(self, frame):
        """
        This method will be overridden by subclasses to perform custom processing.
        """
        return frame

    def cleanup(self):
        """
        Cleans up the camera resources.
        """
        self.running = False
        if self.cap:
            logging.info("Releasing camera resources")
            self.cap.close()
            self.cap = None


