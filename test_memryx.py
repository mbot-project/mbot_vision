from flask import Flask, Response
from utils.utils import register_signal_handlers
from camera.camera import Camera
from memryx import AsyncAccl
from utils.config import CAMERA_CONFIG
import numpy as np
import cv2

# Define class names for your cone classes
CLASS_NAMES = [
    'blue_cone',
    'green_cone',
    'pink_cone',
    'red_cone',
    'yellow_cone'
]

class MemryxCamera(Camera):
    def __init__(self, camera_id, width, height, model, fps=None, confidence_thres=0.8, iou_thres=0.6):
        super().__init__(camera_id, width, height, fps)
        self.model = model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.input_width = 640  # Set input width
        self.input_height = 640  # Set input height

    def capture_and_preprocess(self):
        frame = self.capture_frame()
        # Detailed preprocessing to match yolov8.py
        original_img = frame
        img_height, img_width, _ = original_img.shape
        self.length = max(img_height, img_width)  # Store length as an instance attribute
        image = np.zeros((self.length, self.length, 3), np.uint8)
        image[0:img_height, 0:img_width] = original_img

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)
        blob = blob.squeeze(0)  # Removes the batch dimension -> (3, 640, 640)
        blob = blob.transpose(1, 2, 0)  # Change to (640, 640, 3)
        blob = np.expand_dims(blob, axis=2)  # Add new axis at index 2 -> (640, 640, 1, 3)
        return blob

    def postprocess_and_print(self, *mxa_output):
        """
        Post-processes the model output using the post-processing model.
        The post-processing model handles the output transformation.
        """
        # Get the processed output from the post-processing model
        outputs = np.transpose(np.squeeze(mxa_output[0]))

        # Extract boxes and scores
        boxes = outputs[:, :4]  # x_center, y_center, width, height
        class_scores = outputs[:, 4:]  # 5 cone classes
        # Get confidence scores and class IDs
        confidence = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidence >= self.confidence_thres
        if not np.any(mask):
            print("No detections above confidence threshold")
            return []
            
        # Apply confidence mask
        boxes = boxes[mask]
        confidence = confidence[mask]
        class_ids = class_ids[mask]
        
        # Convert normalized coordinates to pixel coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height
        
        # Convert box coordinates to corner format
        boxes_corner = np.zeros_like(boxes)
        boxes_corner[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * x_factor  # left
        boxes_corner[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * y_factor  # top
        boxes_corner[:, 2] = boxes[:, 2] * x_factor  # width
        boxes_corner[:, 3] = boxes[:, 3] * y_factor  # height
        
        # Create detection dictionaries
        detections = [{
            'bbox': boxes_corner[i].astype(int).tolist(),
            'class_id': int(class_ids[i]),
            'class': CLASS_NAMES[int(class_ids[i])],
            'score': float(confidence[i])
        } for i in range(len(boxes_corner))]
        
        # Apply NMS
        if detections:
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, 
                scores_for_nms, 
                self.confidence_thres, 
                self.iou_thres
            )
            
            if len(indices) > 0:
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []
            
        # Print final detections
        print("\nFinal Detections after NMS:")
        for det in final_detections:
            print(f"Class: {det['class']}, Score: {det['score']:.4f}, BBox: {det['bbox']}")
            
        return final_detections

def main():
    try:
        # Connect the accelerator
        accl.connect_input(camera.capture_and_preprocess)
        accl.connect_output(camera.postprocess_and_print)

        # Run the accelerator
        accl.wait()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
    finally:
        # Cleanup
        camera.cleanup()
        print("Camera resources released")

# # Create Flask app
# app = Flask(__name__)

# @app.route('/')
# def video():
#     return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Update main function to run Flask app
if __name__ == '__main__':
    # Setup camera configuration
    config = CAMERA_CONFIG
    camera_id = config["camera_id"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    fps = config["fps"]

    try:
        # Initialize MemryX accelerator
        accl = AsyncAccl(dfp='utils/yolov8n/yolo8n.dfp')
        
        # Set the post-processing model
        accl.set_postprocessing_model('utils/yolov8n/best_v8n_post.onnx')

        # Initialize camera with MemryxCamera
        camera = MemryxCamera(camera_id=camera_id, width=image_width, height=image_height, model=accl, fps=fps)
        register_signal_handlers(camera.cleanup)
        
        main()
    except KeyboardInterrupt:
        print("Program terminated by user")
    # app.run(host='0.0.0.0', port=5001) 