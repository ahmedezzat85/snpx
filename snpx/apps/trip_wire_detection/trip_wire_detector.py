import os
import cv2
import numpy as np
from time import time, sleep
from datetime import datetime

from snpx import utils
from .. obj_detector import snpx_object_detector

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

def line_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + 1e-15)
  
def in_frange(f, float1, float2):
    """ Evaluate the expression `float1 =< f <= float2`."""
    fmin = min(float1, float2)
    fmax = max(float1, float2)
    return (fmin <= f <= fmax)

def is_rect_intersected(rect1, rect2):
    """ Check the intersection of two rectangles."""
    def _sort(a,b):
        return (a, b) if a < b else (b, a)

    (x1_a, y1_a), (x1_b, y1_b) = rect1
    (x2_a, y2_a), (x2_b, y2_b) = rect2
    x1_min, x1_max = _sort(x1_a, x1_b)
    x2_min, x2_max = _sort(x2_a, x2_b)
    y1_min, y1_max = _sort(y1_a, y1_b)
    y2_min, y2_max = _sort(y2_a, y2_b)

    if (x1_min > x2_max) or (x2_min > x1_max): return False
    if (y1_min > y2_max) or (y2_min > y1_max): return False
    return True

class TripWireDetector(object):
    """ """
    def __init__(self, camera, platform='', object_detection_model='', enable_recording=False):
        self.cam      = camera
        self.detector = snpx_object_detector(platform, self.cam, object_detection_model, self)
        self.logger   = utils.create_logger('twd')

        # Line settings
        self.tw_end   = (0,0)
        self.tw_start = (0,0)
        self.tw_slope = 0
        self.tw_color = (0, 200, 0)
        self.beep_on  = False

        # Create video writer for recording
        self.recording_on = enable_recording
        if enable_recording is True:
            w, h = self.cam.resolution
            self.video_writer = cv2.VideoWriter('twd.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (w, h))

    def start(self):
        """ """
        # Setup the camera fixed scene
        self.cam.setup_scene()

        # Draw the trip-wire
        self._draw_trip_wire()
        self.tw_slope = line_slope(*self.tw_start, *self.tw_end)

        # Start Object Detector
        self.detector.start()

        # Stop Object Detector
        self.detector.close()

        # Close the Video Writer
        if self.recording_on is True:
            self.video_writer.release()
    
    def __call__(self, frame, bboxes):
        """ """
        if utils.Esc_key_pressed():
            self.detector.stop()
            return
        
        beep_alarm = False
        for box in bboxes:
            box_color = (0, 255, 0) # GREEN
            if self._is_event(box) == True:
                beep_alarm = True
                box_color = (0,0,255) # RED
            self.detector.draw_bbox(frame, box, box_color)
        cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 3)
        cv2.imshow(self.cam.name, frame)
        if self.recording_on is True:
            self.video_writer.write(frame)

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

    def _draw_trip_wire(self):
        """ """
        # Mouse Event Callback
        def mouse_cb(event, x, y, flags, param):
            curr_pt = (x, y)
            # Start Drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing  = True
                self.tw_start = curr_pt
            # Show the line while drawing
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing is True:
                cv2.line(self.frame, self.tw_start, curr_pt, self.tw_color, 5)
            # Drawing ends
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.tw_end  = curr_pt
                cv2.line(self.frame, self.tw_start, self.tw_end, self.tw_color, 5)

        # Draw the trip-wire
        self.drawing = False
        self.frame = self.cam.read()
        h, w, c = self.frame.shape
        cv2.putText(self.frame, 'Draw the Trip-Wire and press Esc', (10, h//2), 3, 0.5, (0, 255, 0), 1)
        cv2.setMouseCallback(self.cam.name, mouse_cb)
        while True:
            cv2.imshow(self.cam.name, self.frame)
            if utils.Esc_key_pressed(): break
        # Always set start point to lower y coordinate
        if self.tw_end[1] < self.tw_start[1]:
            tmp = self.tw_start
            self.tw_start = self.tw_end
            self.tw_end   = tmp

        frame = self.cam.read()
        cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 2)
        cv2.imshow(self.cam.name, frame)

    def _is_event(self, box, rule='bidirectional'):
        """ """
        is_event = False
        _, x1, x2, y1, y2, __ = box
        tw_xl, tw_y1 = self.tw_start
        s11 = line_slope(tw_xl, tw_y1, x1, y1)
        s12 = line_slope(tw_xl, tw_y1, x1, y2)
        s21 = line_slope(tw_xl, tw_y1, x2, y1)
        s22 = line_slope(tw_xl, tw_y1, x2, y2)
        if rule.lower() == 'bidirectional':
            if is_rect_intersected((self.tw_start, self.tw_end), ((x1, y1), (x2, y2))) is True:
                if in_frange(self.tw_slope, s11, s12) or in_frange(self.tw_slope, s11, s21) or \
                   in_frange(self.tw_slope, s22, s12) or in_frange(self.tw_slope, s22, s21): 
                   is_event = True

        elif rule.lower() == 'to_right': pass
        elif rule.lower() == 'to_left' : pass
        return is_event