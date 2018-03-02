from __future__ import absolute_import

import os
import cv2
import logging
from . imagenet_downloader import ImageNetDownloader

class DictToAttrs(object):
    def __init__(self, d):
        self.__dict__ = d

def create_dir(dir_path):
    """ Create a directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    if not os.path.isdir(dir_path):
        raise ValueError("Cannot Create Directory %s", dir_path)

def create_logger(name='logger', log_file=None, mode='w'):
    """ Create a Logger Instance."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    ## Add a console handler
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(logging.Formatter(fmt="(%(name)s)[%(levelname)s]%(message)s"))
    logger.addHandler(hdlr)

    ## Add a file handler
    if log_file is not None:
        hdlr = logging.FileHandler(filename=log_file, mode=mode)
        hdlr.setFormatter(logging.Formatter(fmt="%(message)s"))
        logger.addHandler(hdlr)
    
    logger.propagate = False
    return logger

def list_split(L, num_chunks):
    list_sz = len(L)
    chunk_sz = list_sz // num_chunks

    chunk_list = []
    for i in range(num_chunks - 1):
        chunk = L[i*chunk_sz : (i+1)* chunk_sz]
        chunk_list.append(chunk)
    chunk_list.append(L[(num_chunks - 1) * chunk_sz :])
    return chunk_list

def download_images(wnids):
    """ """
    downloader = ImageNetDownloader(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    for id in wnids:
        list = downloader.getImageURLsOfWnid(id)
        downloader.downloadImagesByURLs(id, list)

if __name__ == '__main__':
    download_images(['n08505018'])