from __future__ import absolute_import

import os
import cv2
import numpy as np
from imutils.video import FPS
from snpx.utils import Esc_key_pressed

class OpencvDetector(object):
    """ """
    def __init__(self, model, detection_cb=None):
        prototxt    = model.model_prfx + '.prototxt'
        weights     = model.model_prfx + '.caffemodel'
        self.net    = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.model  = model
        self.det_cb = detection_cb
        self.fps    = FPS().start()

    def __call__(self, frame):
        """ Process a frame from camera. The frame is queued in the Frame FIFO for 
        later processing through the detection_task.
        
        Parameters
        ----------
        frame: numpy.ndarray
            Captured frame from Camera Device.
        
        Returns
        -------
        A boolean flag whether the capturing is stopped or not.
        """
        if Esc_key_pressed():
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))
            return True

        img = self.model.preprocess(frame, resize_only=True)
        img = cv2.dnn.blobFromImage(img, self.model.scale, self.model.in_size, self.model.mean, False)
        self.net.setInput(img)
        net_out = self.net.forward()
        bboxes  = self.model.postprocess(frame, net_out)
        self.det_cb(frame, bboxes)
        self.fps.update()
        return False

    def close(self):
        pass
