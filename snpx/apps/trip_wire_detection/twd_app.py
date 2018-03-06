import os
import argparse
import cv2
import numpy as np
import logging

from time import time, sleep
from datetime import datetime
import utils

from snpx_obj_detector import SNPXObjectDetector

CURR_DIR = os.path.dirname(__file__)
BEEP_SOUND_FILE = os.path.join(CURR_DIR, 'beep.wav')

try:
    import winsound
    def play_sound(flag):
        if flag is True:
            winsound.PlaySound(BEEP_SOUND_FILE, winsound.SND_ASYNC | winsound.SND_LOOP)
        else:
            winsound.PlaySound(None, winsound.SND_ASYNC)
except ImportError:
    def play_sound(flag):
        pass
        
class TripWireDrawer(object):
    """ """
    def __init__(self, winname, image):
        self.done       = False
        self.drawing    = False
        self.start_pt   = (0,0)
        self.end_pt     = (0,0)
        self.color      = (0,192,0)
        self.line_width = 5
        self.frame      = image
        self.winname    = winname
 
    def draw_line(self, event, curr_pt):
        """ """
        if self.done == True:
            return
        # Start Drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = curr_pt
        # Show the line while drawing
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.frame, self.start_pt, curr_pt, self.color, self.line_width)
        # Drawing ends
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt  = curr_pt
            cv2.line(self.frame, self.start_pt, self.end_pt, self.color, self.line_width)

    def __call__(self):
        """ """
        def mouse_cb(event, x, y, flags, param):
            self.draw_line(event, (x,y))

        cv2.setMouseCallback(self.winname, mouse_cb)
        while True:
            cv2.imshow(self.winname, self.frame)
            if utils.Esc_key_pressed():
                done = True
                break
        # Always set start point to lower y coordinate
        if self.end_pt[1] < self.start_pt[1]:
            tmp = self.start_pt
            self.start_pt = self.end_pt
            self.end_pt   = tmp
        return self.start_pt, self.end_pt

def draw_trip_wire(cam_hdl, tripwire_color):
    """ """
    # Draw the trip-wire
    frame = cam_hdl.get_frame()
    h, w, c = frame.shape
    cv2.putText(frame, 'Draw the trip-wire and press Esc', 
                (10, h//2), 3, 0.5, (0, 255, 0), 1)
    drawer = TripWireDrawer(cam_hdl.name, frame)
    tw_start, tw_end = drawer()
    frame = cam_hdl.get_frame()
    cv2.line(frame, tw_start, tw_end, tripwire_color, 2)
    cv2.imshow(cam_hdl.name, frame)
    return tw_start, tw_end

class TWD_Demo(object):
    """ """
    def __init__(self, cam_id=None, ip_cam=None, cam_res=(640, 480)):
        if cam_id == None and ip_cam is None:
            raise ValueError('No Camera given')

        win_name='Trip-Wire Detection'
        if cam_id is not None:
            self.cam = utils.get_default_cam(cam_id=cam_id, win_name=win_name, resolution=cam_res)
        else:
            self.cam = utils.get_default_cam('ip-cam', cam_url=ip_cam, win_name=win_name)

        self.beep_on  = False
        self.tw_start = (0,0)
        self.tw_end   = (0,0)
        self.tw_slope = 0
        self.tw_color = (0, 200, 0)
        self._create_logger()

    def _create_logger(self):
        """ Create a Logger Instance."""
        # Create a new logger instance
        self.logger = logging.getLogger('TWD')
        self.logger.setLevel(logging.WARN)
        
        ## Add a console handler
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(logging.Formatter(fmt="(%(name)s)[%(levelname)s]%(message)s"))
        self.logger.addHandler(hdlr)
        self.logger.propagate = False

    def _slope(self, end_pt):
        x1, y1 = self.tw_start
        x2, y2 = end_pt
        d = (x2 - x1)
        if d == 0:
            d += 1e-8
        return (y2 - y1) / d
        
    def is_event_detected(self, box, rule):
        """ """
        event_detected = False
        _, obj_x1, obj_x2, obj_y1, obj_y2, __ = box
        tw_xl, tw_y1 = self.tw_start
        s11 = self._slope((obj_x1, obj_y1))
        s12 = self._slope((obj_x1, obj_y2))
        s21 = self._slope((obj_x2, obj_y1))
        s22 = self._slope((obj_x2, obj_y2))
        if rule.lower() == 'bidirectional':
            if utils.is_rect_intersected((self.tw_start, self.tw_end), 
                                         ((obj_x1, obj_y1), (obj_x2, obj_y2))) is True:
                if utils.in_frange(self.tw_slope, s11, s12) or \
                   utils.in_frange(self.tw_slope, s11, s21) or \
                   utils.in_frange(self.tw_slope, s22, s12) or \
                   utils.in_frange(self.tw_slope, s22, s21): event_detected = True

        elif rule.lower() == 'to_right':
            if (obj_x2 >= tw_xl): event_detected = True
        elif rule.lower() == 'to_left':
            if (obj_x1 <= tw_xl): event_detected = True

        self.logger.info('SLOPES  %.2f %.2f %.2f %.2f %.2f', s11, s12, s21, s22, self.tw_slope)
        self.logger.info('BOX     (%d,%d) , (%d,%d)', obj_x1, obj_y1, obj_x2, obj_y2)
        self.logger.info('Start Point  (%d,%d)', tw_xl, tw_y1)
        self.logger.info('TRUE' if event_detected is True else 'FALSE')
        return event_detected
    
    def run(self, detector, model, framework, enable_threading=False, rule='bidirectional'):
        """ """
        def twd_cb(frame, bboxes):
            stop_flag = utils.Esc_key_pressed()
            beep_alarm = False
            for box in bboxes:
                box_color = (0, 255, 0) # GREEN
                if self.is_event_detected(box, rule) == True:
                    beep_alarm = True
                    box_color = (0,0,255) # RED
                # Always show the Trip-Wire line on each frame 
                utils.draw_box(frame, box, box_color)
            cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 3)
            cv2.imshow(self.cam.name, frame)
            self.writer.write(frame)

            # Play a beep Sound if the monitored event is detected
            if beep_alarm == True:
                beep_alarm = False
                if self.beep_on == False:
                    play_sound(True)
                    self.beep_on = True
            else:
                if self.beep_on == True:
                    play_sound(False)
                    self.beep_on = False
            return stop_flag
        
        fourcc        = cv2.VideoWriter_fourcc(*'XVID')
        vid_out_file  = 'twd.avi'
        self.writer   = cv2.VideoWriter(vid_out_file, fourcc, 9, (480, 640))
        self.cam.setup_scene()
        self.tw_start, self.tw_end = draw_trip_wire(self.cam, self.tw_color)
        self.tw_slope = self._slope(self.tw_end)
        self.detector = SNPXObjectDetector(detector, model, framework)
        self.detector.set_cam(self.cam)
        self.detector.start(use_threads=enable_threading, detection_cb=twd_cb)
        self.detector.stop()
        self.writer.release()


def main():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, default='')
    parser.add_argument('-d', '--detector', type=str, default='yolo')
    parser.add_argument('-m', '--model', type=str, default='tiny-yolo-v2')
    parser.add_argument('-f', '--framework', type=str, default='opencv-caffe')
    parser.add_argument('-t', '--threading', default=False)
    args = parser.parse_args()

    cam_id = args.video if args.video else 0
    demo = TWD_Demo(cam_id)
    demo.run(args.detector, args.model, args.framework, enable_threading=args.threading)
    
if __name__ == "__main__":
    main()