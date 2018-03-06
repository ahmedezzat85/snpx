import cv2
import os
import sys
from time import time, sleep
from datetime import datetime
import argparse
from imutils.video import FPS
import utils
from threading import Thread
from queue import Queue

from detector.ssd.ssd import SSDObjectDetector
from detector.yolo.yolo import YoloObjectDetector

def get_obj_detector(detector, framework):
    """ """
    if detector.lower() == 'yolo':
        det = YoloObjectDetector(framework=framework)
    else:
        det = SSDObjectDetector(framework=framework)
    return det

class LiveDetector(object):
    """
    """
    def __init__(self, detector, cam_id):
        self.detector = detector
        if isinstance(cam_id, str):
           root , self.out_file = os.path.split(cam_id)
        else:
            self.out_file = 'out.avi'
        self.cam      = utils.WebCam(cam_id, 'Object Detector App')
        self.fps      = FPS().start()
        self.in_q     = Queue()
        self.frame_q  = Queue()
        self.out_q    = Queue()
        self.writer   = None
        self.stopped  = False
        self.threads  = []

    def start(self):
        for fn in [self.preprocess, self.process, self.posprocess]:
         th = Thread(target=fn)
         th.start()
         self.threads.append(th)

    def preprocess(self):
        while True:
            if self.stopped is True: break
            frame = self.cam.get_frame()
            self.frame_q.put(frame)
            if frame is None: 
                self.in_q.put(None)
                break
            preprocessed = self.detector.preprocess(frame)
            self.in_q.put(preprocessed)
        print ('Thread 1 Exited')

    def process(self):
        while True:
            if self.stopped is True: break
            net_in = self.in_q.get()
            if net_in is None:
                self.out_q.put(None)
                self.frame_q.put(None)
                break
            self.fps.update()
            det_out = self.detector.detect(net_in)
            self.out_q.put(det_out)
            self.in_q.task_done()
        print ('Thread 2 Exited')

    def posprocess(self):
        while True:
            if self.stopped is True: break
            frame   = self.frame_q.get()
            net_out = self.out_q.get()
            if frame is None and net_out is None:
                break
            if net_out is not None:
                bboxes = self.detector.get_bboxes(frame, net_out)
                utils.draw_detections(frame, bboxes)

            if self.writer is None:
                fourcc  = cv2.VideoWriter_fourcc(*'XVID')
                h, w, c = frame.shape
                self.writer  = cv2.VideoWriter(self.out_file, fourcc, self.cam.fps, (w, h))
            
            cv2.imshow(self.cam.name, frame)
            self.writer.write(frame)

            self.frame_q.task_done()
            self.out_q.task_done()
        print ('Thread 3 Exited')
        self.stopped = True

    def stop(self):
        print ('stop')
        self.stopped = True
        self.in_q.put(None)
        self.out_q.put(None)
        self.frame_q.put(None)
        for th in self.threads:
            th.join()

        self.fps.stop()
        print("Elapsed    = {:.2f} sec".format(self.fps.elapsed()))
        print("Frame Rate = {:.2f} fps".format(self.fps.fps()))    
        self.cam.close()
        self.writer.release()


def main():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', type=str, default='yolo')
    parser.add_argument('-f', '--framework', type=str, default='opencv-caffe')
    parser.add_argument('-v', '--video', type=str, default='')
    args = vars(parser.parse_args())

    framework  = args['framework']
    detector   = args['detector']
    det        = get_obj_detector(detector, framework)

    if args['video']:
        video = args['video']
    else:
        video = 0

    streamer = LiveDetector(det, video)
    streamer.start()
    while True:
        if utils.Esc_key_pressed(): break
        if streamer.stopped == True: 
            print ('STOPPED')
            break
    streamer.stop()
    
if __name__ == '__main__':
    main()
