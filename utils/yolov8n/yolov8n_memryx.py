from pathlib import Path
import numpy as np
import cv2
from utils.config import CAMERA_CONFIG

class YOLOv8n:
    """
    Base YOLOv8n class for Memryx chip implementation.
    Handles pre/post processing steps specific to YOLOv8n architecture.
    """
    def __init__(self, img_size=None):
        self.name = 'yolov8n'
        self.input_size = (640, 640, 3)
        # Pre-calculate ratio/pad values for preprocessing
        config = CAMERA_CONFIG
        image_width = config["image_width"]
        image_height = config["image_height"]

        # Pre-calculate ratio/pad values for preprocessing
        self.stream_mode = False
        self.preprocess(np.zeros((image_height, image_width, 3)))
        self.stream_mode = True

    def preprocess(self, img):
        h0, w0 = img.shape[:2] # orig hw

        r = self.input_size[0] / max(h0, w0)  # resize img to img_size
        if r != 1:  
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        img, ratio, dwdh = self._letterbox(img, new_shape=self.input_size, auto=False)
        img = img.astype(np.float32)
        img /= 255.0 # Scale

        shapes = (h0, w0), ((h / h0, w / w0), dwdh)

        if not self.stream_mode:
            self.ratio = r
            self.pad = dwdh

        return img
    
    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32):
        """
        A letterbox function.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, r, (dw, dh)