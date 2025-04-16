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
Simple script to test YOLOv8n model with MX3 acceleration.
"""

import sys
import os
import signal
import atexit
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MX3_Test")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import detector
from utils.yolov8n.yolov8n_mxa_detector import Yolov8nMxaDetector

# Global reference to detector
detector = None

def cleanup_resources():
    """Clean up resources on exit."""
    global detector
    if detector is not None:
        try:
            detector.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals."""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup_resources()
    sys.exit(0)

def check_requirements():
    """Check if system meets requirements for running the detector."""
    # Check MX3 support
    try:
        import memryx
    except ImportError:
        logger.error("Memryx library not found - please install it")
        return False
        
    # Check for model files
    dfp_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/yolo8n.dfp'
    post_model_path = '/home/mbot/mbot_ws/mbot_vision/utils/yolov8n/best_v8n_post.onnx'
    
    if not os.path.exists(dfp_path):
        logger.error(f"DFP file not found: {dfp_path}")
        return False
    logger.info(f"Found DFP file: {dfp_path}")
        
    if not os.path.exists(post_model_path):
        logger.error(f"Post-processing model not found: {post_model_path}")
        return False
    logger.info(f"Found post-processing model: {post_model_path}")
    
    # Check if camera is available
    try:
        from picamera2 import Picamera2
        cameras = Picamera2.global_camera_info()
        if not cameras:
            logger.error("No cameras detected")
            return False
        logger.info(f"Found {len(cameras)} camera(s)")
    except Exception as e:
        logger.error(f"Error checking camera: {e}")
        return False
        
    return True

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    # Print header
    print("\n" + "="*60)
    print(" "*20 + "YOLOv8n MX3 DETECTOR")
    print("="*60 + "\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Camera configuration
    camera_id = 0
    width = 1280
    height = 720
    fps = 20
    show_display = True
    
    try:
        # Initialize detector
        logger.info(f"Initializing detector with camera ID: {camera_id}")
        detector = Yolov8nMxaDetector(
            camera_id=camera_id,
            width=width, 
            height=height, 
            fps=fps, 
            show=show_display
        )
        
        # Run detector
        logger.info("Starting detector...")
        print("\nRunning - Press Ctrl+C to exit\n")
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        cleanup_resources()
        logger.info("Test completed") 