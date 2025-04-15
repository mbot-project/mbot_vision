from pathlib import Path
import numpy as np
import cv2

class YOLOv8n:
    """
    Base YOLOv8n class for Memryx chip implementation.
    Handles pre/post processing steps specific to YOLOv8n architecture.
    """
    def __init__(self, img_size=None):
        self.models_dir = Path(__file__).resolve().parent
        self.set_model_params()
        if img_size:
            self.scale = min(self.input_size[0] / float(img_size[1]),
                    self.input_size[1] / float(img_size[0]))
        else:
            self.scale = 1

    def set_model_params(self):
        """Set YOLOv8n specific parameters"""
        self.name = 'yolov8n'
        self.input_size = (640, 640, 3)
        self.dfp_path = str(self.models_dir / 'yolo8n.dfp')
        self.post_model = str(self.models_dir / 'best_v8n_post.onnx')

    def preprocess(self, img):
        """Preprocess image for Memryx chip inference"""
        # First resize the image to 640x640 while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(self.input_size[0] / float(h), self.input_size[1] / float(w))
        new_h, new_w = int(h * scale), int(w * scale)
        self.scale = scale  # Store scale for postprocessing
        
        # Resize the image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a padded image of 640x640
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3),
                dtype=np.uint8) * 114
        
        # Place the resized image in the center of the padded image
        start_h = (self.input_size[0] - new_h) // 2
        start_w = (self.input_size[1] - new_w) // 2
        padded_img[start_h:start_h+new_h, start_w:start_w+new_w] = resized_img

        # Return the preprocessed image as float32
        return np.ascontiguousarray(padded_img).astype(np.float32)

    def postprocess(self, fmap):
        """Convert raw detection output to bounding boxes"""
        dets = fmap[0]  # First output contains detections
        
        # Scale boxes back to original image size
        dets[:, :4] /= self.scale
        
        return dets 