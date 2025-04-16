"""
============
Information:
============
Project: YOLOv8n example code on MXA
File Name: yolov8n_mx3.py

============
Description:
============
A script to implement a YOLOv8n model class using the MemryX MX3 accelerator.
"""

# Imports
import numpy as np
import cv2

# Define cone classes - update these with your actual class names
CONE_CLASSES = ["orange_cone", "yellow_cone", "blue_cone", "large_orange_cone"]

class YoloV8n:
    """
    A helper class to run YOLOv8n pre- and post-processing with MemryX.
    """
    def __init__(self):
        """
        The initialization function.
        """
        self.name = 'YoloV8n'
        self.input_size = (640, 640, 3)
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6
        self.class_names = {i: name for i, name in enumerate(CONE_CLASSES)}

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.original_image = img

        # Get the height and width of the input image
        [self.img_height, self.img_width, _] = self.original_image.shape
        
        # Prepare a square image for inference
        self.length = max((self.img_height, self.img_width))
        self.image = np.zeros((self.length, self.length, 3), np.uint8)
        self.image[0:self.img_height, 0:self.img_width] = self.original_image

        # Calculate scale factor
        scale = self.length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

        # Assume 'blob' is currently (1, 3, 640, 640)
        blob = blob.squeeze(0)  # Removes the batch dimension -> (3, 640, 640)
        blob = blob.transpose(1, 2, 0)  # Change to (640, 640, 3)
        blob = np.expand_dims(blob, axis=2)  # Add new axis at index 2 -> (640, 640, 1, 3)

        # Return the preprocessed image data
        return blob

    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8n model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing 
                    'bbox', 'class_id', 'class', and 'score'.
        """
        # Transpose the output to shape (8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height

        # Extract the bounding box information and class scores
        boxes = outputs[:, :4]  # (8400, 4) - x_center, y_center, width, height
        class_scores = outputs[:, 4:]  # (8400, N) - class scores for N classes

        # Find the class with the highest score for each detection
        max_scores = np.max(class_scores, axis=1)  # (8400,) - maximum class score for each detection
        class_ids = np.argmax(class_scores, axis=1)  # (8400,) - index of the best class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]

        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
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

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                # Filter detections based on NMS
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []

        # Return the list of final detections
        return final_detections 