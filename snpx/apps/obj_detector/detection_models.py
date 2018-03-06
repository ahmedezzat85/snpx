from __future__ import absolute_import

import os
import cv2
import numpy as np
from . yolo import yolo_get_candidate_objects
from snpx import PRETRAINED_MODELS_ROOT_DIR, utils

class YoloModel(object):
    """
    """
    def __init__(self, model_name='tiny-yolo-v2'):
        model_name = model_name.lower()
        self.mean  = 0
        self.scale = 0.00392
        if model_name == 'tiny-yolo-v1':
            self.in_size  = (448, 448)
            self.out_size = None
            self.out_node = 'fc9'
        elif model_name == 'tiny-yolo-v2':
            self.in_size  = (416, 416)
            self.out_size = [12, 12, 125]
            self.out_node = 'result'
        self.model_prfx   = os.path.join(_DETECTION_MODELS_ROOT, model_name, model_name)

    def preprocess(self, image, resize_only=False):
        """ """
        h, w = self.in_size
        img = cv2.resize(image, (w, h))
        if resize_only is False:
            img = img * self.scale
        return img

    def postprocess(self, frame, net_out):
        """ """
        if len(net_out.shape) == 4:
            net_out = net_out[0]
        bboxes = yolo_get_candidate_objects(net_out, frame.shape)
        return bboxes

class SSD(object):
    """
    """
    def __init__(self, model_name):
        self.mean       = 127.5
        self.scale      = 0.007843
        self.in_size    = (300, 300)
        self.model_prfx = os.path.join(_DETECTION_MODELS_ROOT, model_name, model_name)
    
    def preprocess(self, image, resize_only=False):
        h, w = self.in_size
        img = cv2.resize(image, (w, h))
        if resize_only is False:
            img -= self.mean
            img = img * self.scale
        return img

    def postprocess(self, frame, net_out):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # loop over the detections
        h, w, _ = frame.shape
        bboxes = list()
        for i in np.arange(0, net_out.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = net_out[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(net_out[0, 0, i, 1])
                box = net_out[0, 0, i, 3:7] * np.array([w, h, w, h])
                (xmin, ymin, xmax, ymax) = box.astype("int")

                # display the prediction
                bboxes.append([CLASSES[idx], xmin, xmax, ymin, ymax, confidence])
                
        return bboxes

_DETECTION_MODELS_ROOT = PRETRAINED_MODELS_ROOT_DIR
_DETECTION_MODELS={\
    'tiny-yolo-v1' : YoloModel,
    'tiny-yolo-v2' : YoloModel,
    'ssd_mobilenet': SSD
}        

def get_model(model_name):
    """ """
    if model_name not in _DETECTION_MODELS:
        raise ValueError('Model %s is not defined', model_name)

    return _DETECTION_MODELS[model_name](model_name)
