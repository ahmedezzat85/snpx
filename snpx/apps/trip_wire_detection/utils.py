import os
import cv2
import numpy as np
import logging
from time import sleep

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
