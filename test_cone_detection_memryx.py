#!/usr/bin/env python3
import numpy as np
import time
from utils.config import CAMERA_CONFIG
from camera.cone_detector_memryx import ConeDetectorMemryx
from camera.camera import Camera

def main():
    # Load camera configuration
    config = CAMERA_CONFIG
    camera_id = config["camera_id"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    fps = config["fps"]

    # Load calibration data
    calibration_data = np.load('cam_calibration_data.npz')

    # Initialize camera
    camera = Camera(camera_id, image_width, image_height, fps)

    try:
        # Initialize cone detector with context manager
        with ConeDetectorMemryx(calibration_data) as cone_detector:
            while True:
                # Get frame from camera
                frame = camera.capture_frame()
                if frame is None:
                    print("Failed to grab frame")
                    break

                # Process frame and detect cones (non-blocking)
                cone_detector.detect_cones(frame)
                
                # Print any available detections
                if cone_detector.detections:
                    print("\n=== New Frame ===")
                    for detection in cone_detector.detections:
                        print(f"Cone detected: X={detection['x_distance']:.2f}mm, Z={detection['z_distance']:.2f}mm, "
                              f"Confidence={detection['confidence']:.2f}")
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        camera.cleanup()

if __name__ == '__main__':
    main() 