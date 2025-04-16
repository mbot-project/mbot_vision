"""
============
Information:
============
Project: YOLOv8n MXA detector
File Name: yolov8n_mxa_detector.py

============
Description:
============
A script to implement a YOLOv8n detector using the MemryX MX3 accelerator.
"""

# Imports
import time
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
from memryx import MultiStreamAsyncAccl
from .yolov8n_mx3 import YoloV8n

class Yolov8nMxaDetector:
    """
    A class to run YOLOv8n on the MemryX MXA for cone detection.
    """
    def __init__(self, video_source='/dev/video0', show=True):
        """
        Initialization function.
        """
        # Display control and stream initialization
        self.show = show
        self.done = False
        self.video_source = video_source

        # Hardcoded model paths
        self.dfp_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/yolo8n.dfp'
        self.post_model_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/best_v8n_post.onnx'

        # Initialize video capture
        self.stream = cv2.VideoCapture(video_source)
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        self.model = YoloV8n()

        # Get frame dimensions
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dims = (self.width, self.height)
        
        # Timing and FPS related
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros(30)
        self.color_wheel = np.random.randint(0, 255, (20, 3)).astype(np.int32)

        # Start display thread
        self.display_thread = Thread(target=self.display)

    def run(self):
        """
        Start inference on the MXA.
        """
        accl = MultiStreamAsyncAccl(dfp=self.dfp_path)  # Initialize the accelerator with DFP
        print("YOLOv8n inference on MX3 started")
        accl.set_postprocessing_model(self.post_model_path, model_idx=0)  # Set the post-processing model

        self.display_thread.start()  # Start the display thread

        start_time = time.time()

        # Connect input and output streams for the accelerator
        accl.connect_streams(self.capture_and_preprocess, self.postprocess, 1)
        accl.wait()

        self.done = True

        # Join display thread
        self.display_thread.join()

    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame and pre-processes it.
        """
        while True:
            got_frame, frame = self.stream.read()

            if not got_frame:
                return None

            if self.cap_queue.full():
                # drop frame
                continue
            else:
                try:
                    # Put the frame in the cap_queue to be processed later
                    self.cap_queue.put(frame, timeout=2)

                    # Pre-process the frame using the model
                    frame = self.model.preprocess(frame)
                    return frame

                except Full:
                    print('Dropped frame .. exiting')
                    return None

    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the output from MXA.
        """
        dets = self.model.postprocess(mxa_output)  # Get detection results

        # Queue detection results for display
        self.dets_queue.put(dets)

        # Calculate FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index += 1

        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)

        if self.dt_index >= 30:
            self.dt_index = 0

        self.frame_end_time = time.time()

    def display(self):
        """
        Displays the processed frames with detections.
        """
        while not self.done:
            if not self.cap_queue.empty() and not self.dets_queue.empty():
                frame = self.cap_queue.get()
                dets = self.dets_queue.get()

                self.cap_queue.task_done()
                self.dets_queue.task_done()

                # Draw detection boxes
                for d in dets:
                    x1, y1, w, h = d['bbox']
                    color = tuple(int(c) for c in self.color_wheel[d['class_id'] % 20])

                    # Draw bounding boxes
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                    # Add class label and confidence
                    label = f"{d['class']} {d['score']:.2f}"
                    frame = cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add FPS to frame
                fps_text = f"{self.model.name} - {self.fps:.1f} FPS" if self.fps > 1 else self.model.name
                frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Display the frame
                if self.show:
                    cv2.imshow("YOLOv8n MX3 Detector", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                self.done = True

        # Close all windows and release resources
        cv2.destroyAllWindows()
        self.stream.release()

    def get_detections(self):
        """
        Returns the latest detections.
        """
        if not self.dets_queue.empty():
            return self.dets_queue.get()
        return []

    def cleanup(self):
        """
        Cleans up resources.
        """
        self.done = True
        if self.display_thread.is_alive():
            self.display_thread.join()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows() 