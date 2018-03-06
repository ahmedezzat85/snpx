from __future__ import absolute_import

import argparse
from snpx.apps import snpx_camera, snpx_object_detector

def main():
    """ """
    parser = argparse.ArgumentParser('SNPX Object Detection App')
    parser.add_argument('-c', '--camera', type=str, default='')
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('-p', '--platform', type=str, default='')
    parser.add_argument('-r', '--resolution', type=str, default='640x480')
    args = parser.parse_args()

    r1,r2 = args.resolution.split('x')
    resolution = (int(r1), int(r2))
    snpx_cam = snpx_camera(args.camera, 'Synaplexus Object Detection App', resolution)
    snpx_object_detector(args.platform, snpx_cam, args.model)
    
if __name__ == "__main__":
    main()
