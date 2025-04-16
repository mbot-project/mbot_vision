#!/usr/bin/env python3
"""
============
Information:
============
Project: YOLOv8n MX3 Test
File Name: run_mx3_test.py

============
Description:
============
A simple script to test the YOLOv8n model with MX3 acceleration.
"""

import sys
import os
import signal

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.yolov8n.yolov8n_mxa_detector import Yolov8nMxaDetector

def signal_handler(sig, frame):
    """
    Handle Ctrl+C to gracefully exit the program.
    """
    print("\nExiting...")
    if 'detector' in globals():
        detector.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up video source - you can change this to your camera device or a video file
    video_source = '/dev/video0'  # Default camera
    
    # Initialize the detector
    print(f"Initializing YOLOv8n MX3 detector with video source: {video_source}")
    detector = Yolov8nMxaDetector(video_source=video_source, show=True)
    
    try:
        # Run the detector
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        detector.cleanup()
        print("Testing completed.") 