from __future__ import absolute_import

import os
import cv2
import numpy as np
from time import time, sleep
from datetime import datetime
from snpx import utils
from . detection_models import get_model
from . mvncs_detector import MvNCSDetector
from . opencv_detector import OpencvDetector

class SNPXObjectDetector(object):
    """ Abstraction for Object Detector.

    Parameters
    ----------
    camera: snpx.Camera
        Camera Instance.
    detector_model_name: str
        Name of the object detection neural network model.
    draw_detections: bool
        Whether to draw bounding boxes on detected objects or not. 
    """
    def __init__(self, camera, detector_model_name, platform='mvncs', det_cb=None):
        self.cam      = camera
        self.model    = get_model(detector_model_name)
        if det_cb is None: det_cb = self.default_cb
        if platform == 'mvncs':
            self.detector = MvNCSDetector(self.model, det_cb)
        elif platform == 'opencv':
            self.detector = OpencvDetector(self.model, det_cb)

    def start(self):
        """ """
        self.cam.start(self.detector)

    def stop(self):
        """ """
        self.cam.close()
        self.detector.close()

    def default_cb(self, frame, bboxes):
        self.draw_bounding_boxes(frame, bboxes)
        cv2.imshow(self.cam.name, frame)
        cv2.waitKey(10)
        return False
        
    def draw_box(self, img, box, box_color=(0, 255, 0)):
        """ draw a single bounding box on the image """
        name, x_start, x_end, y_start, y_end, score = box
        h, w, _ = img.shape
        font = (1e-3 * h) * 0.5
        thick = int((h + w) // 300)
        box_tag = '{} : {: .2f}'.format(name, score)
        text_x, text_y = 5, 7

        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), box_color, thick)
        boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x_start, y_start-boxsize[1]-text_y),
                    (x_start+boxsize[0]+text_x, y_start), box_color, -1)
        cv2.putText(img, box_tag, (x_start+text_x, y_start-text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick//3)

    def draw_bounding_boxes(self, img, bboxes):
        for bbox in bboxes:
            self.draw_box(img, bbox)
