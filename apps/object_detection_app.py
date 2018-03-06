from __future__ import absolute_import

import argparse
from snpx.apps import snpx_camera
from snpx.apps import SNPXObjectDetector

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
    detector = SNPXObjectDetector(snpx_cam, args.model, args.platform)
    detector.start()
    detector.stop()
    
if __name__ == "__main__":
    main()
