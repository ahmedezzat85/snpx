from __future__ import absolute_import

from . detector import SNPXMvNCSDetector, SNPXOpenCVDetector

def snpx_object_detector(platform, camera, model_name, det_cb=None):
    if platform == 'mvncs':
        detector = SNPXMvNCSDetector(camera, model_name, det_cb)
    elif platform == 'opencv':
        detector = SNPXOpenCVDetector(camera, model_name, det_cb)
    else:
        raise ValueError('Unknown Platform for object detection %s', platform)

    detector.start()
    detector.close()
