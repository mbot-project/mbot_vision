"""
============
Information:
============
Project: YOLOv8n model class for MX3 accelerator
File Name: yolov8n_mx3.py

============
Description:
============
Defines the YOLOv8n model class that handles pre- and post-processing
for inference using the MemryX MX3 accelerator.
"""

import numpy as np
import cv2
import logging

# Configure logging - only show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLOv8n_Model")

# Define cone classes
CONE_CLASSES = ["red_cone", "yellow_cone", "blue_cone", "green_cone", "pink_cone"]

class YoloV8n:
    """
    Helper class for YOLOv8n pre- and post-processing with MemryX.
    """
    def __init__(self):
        """Initialize model parameters."""
        self.name = 'YoloV8n'
        self.input_size = (640, 640, 3)
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.6
        self.iou_thres = 0.6
        self.class_names = {i: name for i, name in enumerate(CONE_CLASSES)}
        logger.info(f"Initialized YOLOv8n model with {len(CONE_CLASSES)} classes")

    def preprocess(self, img):
        """
        Preprocess the input image before inference.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Preprocessed image ready for inference
        """
        try:
            if img is None:
                logger.error("Received None image for preprocessing")
                return None
                
            self.original_image = img

            # Get image dimensions and create square image
            [self.img_height, self.img_width, _] = self.original_image.shape
            self.length = max((self.img_height, self.img_width))
            self.image = np.zeros((self.length, self.length, 3), np.uint8)
            self.image[0:self.img_height, 0:self.img_width] = self.original_image

            # Prepare blob for model
            blob = cv2.dnn.blobFromImage(self.image, scalefactor=1/255, size=(640, 640), swapRB=True)

            # Process blob format for MX3
            blob = blob.squeeze(0)  # Removes batch dimension -> (3, 640, 640)
            blob = blob.transpose(1, 2, 0)  # Change to (640, 640, 3)
            blob = np.expand_dims(blob, axis=2)  # Add axis -> (640, 640, 1, 3)

            return blob
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return None

    def postprocess(self, output):
        """
        Process model output to extract bounding boxes, scores, and class IDs.

        Args:
            output: Raw output from the model

        Returns:
            List of detections with bounding boxes, class info, and scores
        """
        try:
            if not output or len(output) == 0:
                return []
                
            # Process the output to the correct shape
            outputs = np.transpose(np.squeeze(output[0]))

            # Calculate scaling factors
            x_factor = self.length / self.input_width
            y_factor = self.length / self.input_height

            # Extract boxes and scores
            boxes = outputs[:, :4]  # (x_center, y_center, width, height)
            class_scores = outputs[:, 4:]  # class scores

            # Get highest scoring class for each detection
            max_scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

            # Filter by confidence threshold
            valid_indices = np.where(max_scores >= self.confidence_thres)[0]
            if len(valid_indices) == 0:
                return []  # No valid detections
            
            # Get valid detections
            valid_boxes = boxes[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            valid_scores = max_scores[valid_indices]

            # Convert box coordinates to (x, y, width, height) format
            valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
            valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
            valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
            valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

            # Create detection dictionaries
            detections = [{
                'bbox': valid_boxes[i].astype(int).tolist(),
                'class_id': int(valid_class_ids[i]),
                'class': self.class_names.get(int(valid_class_ids[i]), "Unknown"),
                'score': valid_scores[i]
            } for i in range(len(valid_indices))]

            # Apply non-maximum suppression
            if len(detections) > 0:
                boxes_for_nms = [d['bbox'] for d in detections]
                scores_for_nms = [d['score'] for d in detections]

                indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, 
                                          self.confidence_thres, self.iou_thres)

                if len(indices) > 0:
                    # Flatten indices if they are returned as a list of arrays
                    if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                        indices = [i[0] for i in indices]

                    # Apply NMS filtering
                    final_detections = [detections[i] for i in indices]
                else:
                    final_detections = []
            else:
                final_detections = []

            return final_detections
            
        except Exception as e:
            logger.error(f"Error during postprocessing: {e}")
            return [] 