from __future__ import absolute_import

import os
import cv2
import numpy as np
from queue import Queue
from threading import Thread
from imutils.video import FPS
from datetime import datetime

from snpx import utils
from .. mvnc_dev import MvNCS

class MvNCSDetector(object):
    """ """
    def __init__(self, model, detection_cb=None):
        # Initializations
        self.model      = model
        self.inp_q      = Queue(2)
        self.det_cb     = detection_cb
        self.bboxes     = None
        self.net_out    = None
        self.stopped    = False
        self.prev_frame = None

        # Open the NCS device
        ncs_graph = model.model_prfx + '.graph'
        self.mvncs = MvNCS(dev_idx=0, dont_block=True)
        self.mvncs.load_model(ncs_graph)

        # Start the detection thread
        Thread(target=self.detection_task).start()

    def process_frame(self, frame, preproc):
        """ """
        # Load frame for Inference
        self.mvncs.load_input(preproc)

        # Process previous Inference
        if self.net_out is not None:
            self.bboxes = self.postprocess(self.prev_frame, self.net_out)
            self.det_cb(self.prev_frame, self.bboxes)
            self.fps.update()

        # Get Inference Result
        self.net_out, _ = self.mvncs.get_output()
        self.prev_frame = frame

    def postprocess(self, frame, net_out):
        """ """
        out_shape = net_out.shape
        if self.model.out_size is not None:
            out_size = self.model.out_size
            net_out = net_out.reshape(out_size)
            net_out = np.transpose(net_out, [2, 0, 1])
        net_out = net_out.astype(np.float32)
        bboxes  = self.model.postprocess(frame, net_out) 
        return bboxes
        
    def detection_task(self):
        """ A python Thread for processing queued frames from camera.
        """
        self.fps = FPS().start()
        while True:
            frame, preproc, skip_frame = self.inp_q.get()
            self.inp_q.task_done()
            if frame is None: break
            if self.stopped is True: break
            self.process_frame(frame, preproc)

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
        if utils.Esc_key_pressed():
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))
            self.stopped = True
            self.inp_q.put((None, None, None))
            return True

        preproc = self.model.preprocess(frame)
        self.inp_q.put((frame, preproc, None))
        return False

    def close(self):
        self.mvncs.unload_model()
        self.mvncs.close()
