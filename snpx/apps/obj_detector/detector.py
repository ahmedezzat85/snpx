from __future__ import absolute_import

import os
import cv2
import numpy as np
from time import time, sleep
from datetime import datetime
from queue import Queue
from threading import Thread
from imutils.video import FPS

from snpx import utils
from . detection_models import get_model
from .. mvnc_dev import MvNCS

def snpx_object_detector(platform, camera, model_name, det_cb=None):
    """ """
    if platform == 'mvncs':
        detector = SNPXMvNCSDetector(camera, model_name, det_cb)
    elif platform == 'opencv':
        detector = SNPXOpenCVDetector(camera, model_name, det_cb)
    else:
        raise ValueError('Unknown Platform for object detection %s', platform)

    return detector

class SNPXBaseDetector(object):
    """ Base class for all Object Detector platforms.

    Parameters
    ----------
    camera: snpx.Camera
        Camera Instance.
    detector_model_name: str
        Name of the object detection neural network model.
    draw_detections: bool
        Whether to draw bounding boxes on detected objects or not. 
    """
    def __init__(self, camera, model_name, det_cb):
        self.cam     = camera
        self.model   = get_model(model_name)
        self.det_cb  = det_cb if det_cb is not None else self._default_cb
        self.stopped = False
        
    def start(self):
        self.fps = FPS().start()
        self.cam.start(self)

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
        if self.stopped is False:
            self.detect(frame)
        return self.stopped

    def detect(self, frame):
        raise NotImplementedError('Must be implemented by Child Classes only')

    def stop(self):
        self.stopped = True
        self.fps.stop()
        print("Elapsed = {:.2f}".format(self.fps.elapsed()))
        print("FPS     = {:.2f}".format(self.fps.fps()))

    def draw_bbox(self, img, box, box_color=(0, 255, 0)):
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

    def _default_cb(self, frame, bboxes):
        for bbox in bboxes:
            self.draw_bbox(frame, bbox)    
        cv2.imshow(self.cam.name, frame)
        cv2.waitKey(1)
        if utils.Esc_key_pressed():
            self.stop()

class SNPXOpenCVDetector(SNPXBaseDetector):
    """ """
    def __init__(self, camera, model_name, det_cb=None):
        super().__init__(camera, model_name, det_cb)
        prototxt    = self.model.model_prfx + '.prototxt'
        weights     = self.model.model_prfx + '.caffemodel'
        self.net    = cv2.dnn.readNetFromCaffe(prototxt, weights)

    def detect(self, frame):
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
        img = self.model.preprocess(frame, resize_only=True)
        img = cv2.dnn.blobFromImage(img, self.model.scale, self.model.in_size, self.model.mean, False)
        self.net.setInput(img)
        net_out = self.net.forward()
        bboxes  = self.model.postprocess(frame, net_out)
        self.det_cb(frame, bboxes)
        self.fps.update()

    def stop(self):
        super().stop()

    def close(self):
        cv2.destroyAllWindows()

class SNPXMvNCSDetector(SNPXBaseDetector):
    """ """
    def __init__(self, camera, model_name, det_cb=None):
        super().__init__(camera, model_name, det_cb)
        self.inp_q      = Queue(2)
        self.bboxes     = None
        self.net_out    = None
        self.prev_frame = None

        # Open the NCS device
        ncs_graph = self.model.model_prfx + '.graph'
        self.mvncs = MvNCS(dev_idx=0, dont_block=True)
        self.mvncs.load_model(ncs_graph)

        # Start the detection thread
        Thread(target=self._detection_task).start()

    def detect(self, frame):
        """ Process a frame from camera. The frame is queued in the Frame FIFO for 
        later processing through the detection_task.
        
        Parameters
        ----------
        frame: numpy.ndarray
            Captured frame from Camera Device.
        """
        if self.stopped is False:
            preproc = self.model.preprocess(frame)
            self.inp_q.put((frame, preproc))

    def stop(self):
        super().stop()

    def close(self):
        self.mvncs.unload_model()
        self.mvncs.close()

    def _process_frame(self, frame, preproc):
        """ """
        # Load frame for Inference
        self.mvncs.load_input(preproc)

        # Process previous Inference
        if self.net_out is not None:
            self.bboxes = self._postprocess(self.prev_frame, self.net_out)
            self.det_cb(self.prev_frame, self.bboxes)
            self.fps.update()

        # Get Inference Result
        self.net_out, _ = self.mvncs.get_output()
        self.prev_frame = frame

    def _postprocess(self, frame, net_out):
        """ """
        out_shape = net_out.shape
        if self.model.out_size is not None:
            out_size = self.model.out_size
            net_out = net_out.reshape(out_size)
            net_out = np.transpose(net_out, [2, 0, 1])
        net_out = net_out.astype(np.float32)
        bboxes  = self.model.postprocess(frame, net_out) 
        return bboxes
        
    def _detection_task(self):
        """ A python Thread for processing queued frames from camera.
        """
        while True:
            frame, preproc = self.inp_q.get()
            self.inp_q.task_done()
            if self.stopped is False:
                self._process_frame(frame, preproc)
            else:
                # Make sure the queue is empty to release any blocked put call
                if self.inp_q.empty():
                    break

