"""
CAMERA
======


"""

import cv2
import numpy as np
from time import sleep
from snpx.utils import Esc_key_pressed, Enter_key_pressed

def snpx_camera(cam_id, win_name, resolution):
    """ Get the right camera object based on the cam_id.

    Parameters
    ----------
    cam_id: str
        An identification for the camera to use. Possible values:
        * `pi-cam` Raspberry Pi Camera.
        * `ip-cam:url` The url is the full address of the ip camera.
        * `video:filename` The filename is a recorded video file.
        * Otherwise it will be an integer identifying an ID for an attached web cam.
    win_name: str
        A title for the live display window.
    resolution: tuple
        Camera capture resolution as a two value tuple. e.g. `(640, 480)`.

    Returns
    -------
    A class instance of the right camera type.
    """
    if cam_id.startswith('pi-cam')  : cam = PiWebCam(win_name, resolution)
    elif cam_id.startswith('ip-cam'): cam = IPCam(cam_id, win_name)
    else: cam = WebCam(cam_id, win_name, resolution)
    return cam

class BaseCamera(object):
    """ Base class for Camera device.
    """
    def __init__(self, win_name):
        self.name  = win_name
        self.setup = True
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    
    def setup_scene(self):
        """ Set the fixed scene of the camera."""
        if self.setup == False: return
        while True:
            frame = self.read()
            h, w, c = frame.shape
            cv2.putText(frame, 'Move the Camera to setup the scene and press Esc', 
                        (10, h//2), 3, 0.5, (0, 255, 0), 1)
            cv2.imshow(self.name, frame)
            if Esc_key_pressed(): break

    def satrt(self):
        """ """
        raise NotImplementedError()

    def close(self):
        """ """
        raise NotImplementedError()

class WebCam(BaseCamera):
    """ Web Camera device capture callback.

    It returns the last captured frame for each call.

    Parameters
    ----------
    cam_id : int
        ID for the attached camera.
    """
    def __init__(self, cam_id, win_name, resolution):
        super().__init__(win_name)
        if cam_id.startswith('video'):
            self.setup = False
            self.fps   = round(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            cam_id = int(cam_id)

        self.cap = cv2.VideoCapture(cam_id)
        if self.cap.isOpened() == False:
            raise ValueError('Camera not opened. Check the camera ID.')
    
    def start(self, capture_cb=None):
        stop_cap = False
        callback = self.default_cb if capture_cb is None else capture_cb
        while True:
            frame = self.read()
            # Pass the frame to the callback
            stop_cap = callback(frame)            
            if stop_cap is True: break
            
    def read(self):
        _, frame = self.cap.read()
        return frame

    def default_cb(self, frame):
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)
        if Esc_key_pressed(): return True
        return False

    def close(self):
        self.cap.release()

class IPCam(BaseCamera):
    """ IP Camera streaming callback.

    It returns the last captured frame for each call.

    Parameters
    ----------
    cam_url : string
        The streaming address of the camera.
    """
    def __init__(self, cam_url, win_name):
        super().__init__(win_name)
        self.url = cam_url
    
    def read(self):
        cap = cv2.VideoCapture(self.url)
        _, frame = cap.read()
        cap.release()
        return frame
    
    def close(self):
        pass

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    class PiWebCam(object):
        def __init__(self, win_name, resolution):
            self.cam = PiCamera()
            self.name = win_name
            self.cam.resolution = resolution
            self.stopped = False
            self.setup   = True
 
        def setup_scene(self):
            """ Set the fixed scene of the camera."""
            def frame_cb(frame):
                stop_cap = Esc_key_pressed()
                h, w, c = frame.shape
                cv2.putText(frame, 'Move the Camera to setup the scene and press Esc', 
                            (10, h//2), 3, 0.5, (0, 255, 0), 1)
                cv2.imshow(self.name, frame)
                return stop_cap

            if self.setup == False: return
            self.start_video_capture(frame_cb)

        def start_video_capture(self, capture_cb=None):
            self.cap = PiRGBArray(self.cam, size=self.cam.resolution)
            sleep(0.1) # allow the camera to warmup
            stop_cap = False
            for pi_frame in self.cam.capture_continuous(self.cap, format="bgr", use_video_port = True):
                frame = pi_frame.array
                # Pass the frame to the callback
                if capture_cb is None:
                    cv2.imshow(self.name, frame)
                else:
                    stop_cap = capture_cb(frame)
                
                self.cap.truncate(0)
                if stop_cap is True: break
            
        def stop(self):
            self.stopped = True

        def get_frame(self):
            self.cam.capture(self.cap, format="bgr", use_video_port=True)
            frame = self.cap.array
            self.cap.truncate(0)
            return frame

        def close(self):
            """ """

except ImportError:
    class PiWebCam(object):
        def __init__(self, win_name, resolution):
            raise NotImplementedError()