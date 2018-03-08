from __future__ import absolute_import

import argparse
from snpx.apps import snpx_camera, snpx_object_detector, TripWireDetector

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
    # twd = TripWireDetector(snpx_cam, args.platform, args.model)
    # twd.start()
    detector = snpx_object_detector(args.platform, snpx_cam, args.model)
    detector.start()
    detector.close()
    
if __name__ == "__main__":
    main()
