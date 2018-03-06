from __future__ import absolute_import

import argparse
from snpx.apps import snpx_camera

def main():
    """ """
    parser = argparse.ArgumentParser('SNPX Object Detection App')
    parser.add_argument('-c', '--camera', type=str, default='')
    parser.add_argument('-r', '--resolution', type=str, default='640x480')
    args = parser.parse_args()

    r1,r2 = args.resolution.split('x')
    resolution = (int(r1), int(r2))
    snpx_cam = snpx_camera(args.camera, 'Synaplexus Object Detection App', resolution)
    snpx_cam.start()
    snpx_cam.close()
    
if __name__ == "__main__":
    main()
