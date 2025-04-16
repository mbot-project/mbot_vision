"""
============
Information:
============
Project: YOLOv8n MXA detector
File Name: yolov8n_mxa_detector.py

============
Description:
============
Implements the YOLOv8n detector using the MemryX MX3 accelerator.
"""

import time
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
import threading
from memryx import MultiStreamAsyncAccl
from yolov8n_mx3 import YoloV8n
from picamera2 import Picamera2
import libcamera
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MXA_Detector")

class Yolov8nMxaDetector:
    """
    A class to run YOLOv8n on the MemryX MXA for cone detection.
    """
    def __init__(self, camera_id=0, width=1280, height=720, fps=20, show=True):
        """
        Initialize the detector with camera and model settings.
        
        Args:
            camera_id: Camera ID to use
            width: Frame width
            height: Frame height
            fps: Frames per second
            show: Whether to display output frames
        """
        # Initialize settings
        self.show = show
        self.done = False
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.accl = None
        
        # Model paths
        self.dfp_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/yolo8n.dfp'
        self.post_model_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/best_v8n_post.onnx'

        # Initialize camera
        self._init_camera()
        
        # Initialize queues, model and metrics
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        self.model = YoloV8n()
        self.frame_count = 0
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps_value = 0
        self.dt_array = np.zeros(30)
        self.color_wheel = np.random.randint(0, 255, (20, 3)).astype(np.int32)
        self.last_successful_frame_time = 0

        # Create display thread
        self.display_thread = Thread(target=self.display)
        self.display_thread.daemon = True

    def _init_camera(self):
        """Initialize and configure the camera."""
        try:
            logger.info("Initializing camera...")
            self.camera = Picamera2(self.camera_id)
            
            # Create and configure preview
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            
            # Set FPS if specified
            if self.fps:
                frame_duration = int((1./self.fps) * 1e6)  # Convert to microseconds
                config["controls"] = {'FrameDurationLimits': (frame_duration, frame_duration)}
            
            # Set camera orientation
            config["transform"] = libcamera.Transform(hflip=1, vflip=1)
            
            # Apply configuration and start camera
            self.camera.align_configuration(config)
            self.camera.configure(config)
            self.camera.start()
            
            # Wait for camera to stabilize
            logger.info("Camera started, waiting for it to stabilize...")
            time.sleep(1.5)
            
            # Verify camera is working with test frame
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Camera initialization failed: Could not capture test frame")
            
            logger.info(f"Camera initialized. Frame size: {test_frame.shape[:2]}")
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera = None
            raise

    def run(self):
        """Start inference on the MX3 accelerator."""
        max_restarts = 2  # Limit number of restart attempts
        restart_count = 0
        
        while restart_count <= max_restarts and not self.done:
            try:
                # Initialize MX3 accelerator
                logger.info(f"Initializing MX3 accelerator (attempt {restart_count+1}/{max_restarts+1})...")
                self.accl = MultiStreamAsyncAccl(dfp=self.dfp_path)
                
                # Set up post-processing model
                self.accl.set_postprocessing_model(self.post_model_path, model_idx=0)
                logger.info("MX3 accelerator initialized")

                # Start display thread if not already running
                if not self.display_thread.is_alive():
                    self.display_thread.start()
                
                # Initialize frame count and timing
                self.frame_count = 0
                self.last_successful_frame_time = time.time()
                
                # Connect to accelerator with timeout handling
                logger.info("Connecting to MX3 accelerator...")
                
                # Set up connection with timeout
                connect_success = threading.Event()
                connect_exception = [None]
                
                def connect_with_timeout():
                    try:
                        self.accl.connect_streams(self.capture_and_preprocess, self.postprocess, 1)
                        connect_success.set()
                    except Exception as e:
                        connect_exception[0] = e
                        connect_success.set()
                
                # Start connection thread
                connect_thread = threading.Thread(target=connect_with_timeout)
                connect_thread.daemon = True
                connect_thread.start()
                
                # Wait for connection
                if not connect_success.wait(timeout=20):
                    logger.error("MX3 connection timed out")
                    restart_count += 1
                    self.accl = None
                    time.sleep(1)
                    continue
                
                # Check for connection errors
                if connect_exception[0] is not None:
                    logger.error(f"Connection error: {connect_exception[0]}")
                    restart_count += 1
                    time.sleep(1)
                    continue
                
                logger.info("Connected to MX3 accelerator")
                
                # Main processing loop
                start_time = time.time()
                max_runtime = 3600  # 1 hour max runtime
                
                # Monitor processing
                while not self.done:
                    # Check for timeouts
                    current_time = time.time()
                    
                    # Check maximum runtime
                    if current_time - start_time > max_runtime:
                        logger.warning("Maximum runtime reached")
                        self.done = True
                        break
                    
                    # Check for long periods without frames
                    elapsed = current_time - self.last_successful_frame_time
                    if elapsed > 20:
                        logger.error("No frames received for too long, restarting")
                        break
                    
                    # Sleep to avoid busy waiting
                    time.sleep(0.1)
                
                # Clean up accelerator
                if self.accl:
                    logger.info("Waiting for MX3 accelerator to finish...")
                    # Use timeout for wait call
                    wait_thread = threading.Thread(target=lambda: self.accl.wait())
                    wait_thread.daemon = True
                    wait_thread.start()
                    wait_thread.join(timeout=5)
                
            except Exception as e:
                logger.error(f"Error during MX3 inference: {e}")
            
            # Prepare for potential restart
            restart_count += 1
            if not self.done and restart_count <= max_restarts:
                logger.info(f"Restarting MX3 accelerator...")
                self.accl = None  # Release current accelerator
                time.sleep(1)
            else:
                break
        
        # Final cleanup
        self.done = True
        if self.display_thread.is_alive():
            self.display_thread.join(timeout=3)

    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame and preprocesses it for the MX3 accelerator.
        
        Args:
            stream_idx: Stream index from the accelerator
            
        Returns:
            Preprocessed frame or None on error
        """
        if self.done:
            return None
        
        # Use retry mechanism for robustness
        max_retries = 2
        
        for retry in range(max_retries):
            try:
                # Capture frame
                frame = self.camera.capture_array()
                
                if frame is None or frame.size == 0:
                    if retry < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    return None
                
                # Update frame stats
                self.frame_count += 1
                self.last_successful_frame_time = time.time()
                
                # Add to display queue if possible
                try:
                    self.cap_queue.put(frame, block=False)
                except Full:
                    pass  # Skip if queue full
                
                # Preprocess frame for accelerator
                preprocessed_frame = self.model.preprocess(frame)
                if preprocessed_frame is None:
                    if retry < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    return None
                
                return preprocessed_frame
                
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                if retry < max_retries - 1:
                    time.sleep(0.2)
        
        return None

    def postprocess(self, stream_idx, *mxa_output):
        """
        Process output from MX3 accelerator.
        
        Args:
            stream_idx: Stream index from the accelerator
            *mxa_output: Output from the accelerator
        """
        try:
            # Skip empty output
            if not mxa_output or len(mxa_output) == 0:
                return
            
            # Get detections
            dets = self.model.postprocess(mxa_output)
            
            # Add to display queue
            if not self.dets_queue.full():
                self.dets_queue.put(dets)
            
            # Calculate FPS
            current_time = time.time()
            if self.frame_end_time > 0:
                self.dt_array[self.dt_index] = current_time - self.frame_end_time
                self.dt_index += 1
                
                if self.dt_index % 15 == 0:
                    self.fps_value = 1 / np.average(self.dt_array)
                
                if self.dt_index >= 30:
                    self.dt_index = 0
            
            self.frame_end_time = current_time
            
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")

    def display(self):
        """Display processed frames with detection results."""
        while not self.done:
            try:
                # Get frame and detections from queues
                if not self.cap_queue.empty() and not self.dets_queue.empty():
                    frame = self.cap_queue.get(timeout=0.1)
                    dets = self.dets_queue.get(timeout=0.1)
                    
                    self.cap_queue.task_done()
                    self.dets_queue.task_done()
                    
                    # Draw detection boxes
                    for d in dets:
                        x1, y1, w, h = d['bbox']
                        color = tuple(int(c) for c in self.color_wheel[d['class_id'] % 20])
                        
                        # Draw bounding box
                        frame = cv2.rectangle(frame, (int(x1), int(y1)), 
                                             (int(x1 + w), int(y1 + h)), color, 2)
                        
                        # Add label with confidence
                        label = f"{d['class']} {d['score']:.2f}"
                        frame = cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Add FPS information
                    fps_text = f"{self.model.name} - {self.fps_value:.1f} FPS" if self.fps_value > 1 else self.model.name
                    frame = cv2.putText(frame, fps_text, (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Display the frame
                    if self.show:
                        cv2.imshow("YOLOv8n MX3 Detector", frame)
                
                # Check for exit key
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # q or ESC
                    self.done = True
                    break
                    
            except Exception as e:
                logger.error(f"Display error: {e}")
            
            # Sleep to avoid CPU overload
            time.sleep(0.01)
        
        cv2.destroyAllWindows()

    def get_detections(self):
        """
        Get the latest detections.
        
        Returns:
            List of current detections
        """
        if not self.dets_queue.empty():
            return self.dets_queue.get()
        return []

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.done = True
        
        # Release accelerator
        if hasattr(self, 'accl') and self.accl is not None:
            try:
                self.accl = None
                logger.info("MX3 accelerator released")
            except Exception as e:
                logger.error(f"Error releasing accelerator: {e}")
        
        # Join display thread
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            try:
                self.display_thread.join(timeout=3)
            except Exception:
                pass
        
        # Release camera
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.close()
                logger.info("Camera released")
            except Exception:
                pass
        
        # Clear queues
        try:
            while not self.cap_queue.empty():
                self.cap_queue.get_nowait()
            while not self.dets_queue.empty():
                self.dets_queue.get_nowait()
        except Exception:
            pass
        
        # Close windows
        cv2.destroyAllWindows() 