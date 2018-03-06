from __future__ import print_function, division

import numpy as np
import cv2
import sys
import os

def get_bboxes(self, image, net_out):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # loop over the detections
    h, w, _ = image.shape
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