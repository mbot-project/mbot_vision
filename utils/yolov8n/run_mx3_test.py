#!/usr/bin/env python3
"""
Test YOLOv8n model with MX3 acceleration.
"""

import sys
import signal
import atexit
import logging
from yolov8n_mxa_detector import Yolov8nMxaDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MX3_Test")

# Global reference to detector
detector = None

def cleanup_resources():
    """Clean up resources on exit."""
    global detector
    if detector is not None:
        try:
            detector.cleanup()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals."""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup_resources()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    # Camera configuration
    camera_id = 0
    width = 1280
    height = 720
    fps = 20
    show_display = True
    
    try:
        # Initialize and run detector
        logger.info(f"Initializing detector with camera ID: {camera_id}")
        detector = Yolov8nMxaDetector(
            camera_id=camera_id,
            width=width, 
            height=height, 
            fps=fps, 
            show=show_display
        )
        
        logger.info("Starting detector...")
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        cleanup_resources() 