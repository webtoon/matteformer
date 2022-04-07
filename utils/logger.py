import os
import cv2
import torch
import logging
import datetime
import numpy as np
from   pprint import pprint
from   utils.config import CONFIG

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def make_color_wheel():
    # from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


class MyLogger(logging.Logger):
    """
    Only write log in the first subprocess
    """
    def __init__(self, *args, **kwargs):
        super(MyLogger, self).__init__(*args, **kwargs)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if CONFIG.local_rank == 0:
            super()._log(level, msg, args, exc_info, extra, stack_info)


def get_logger(log_dir=None, logging_level="DEBUG"):
    """
    Return a default build-in logger if log_file=None and
    Return a build-in logger which dump stdout to log_file if log_file is assigned
    :param log_file: logging file dumped from stdout
    :param logging_level:
    :return: Logger
    """
    level = LEVELS[logging_level.upper()]
    exp_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logging.setLoggerClass(MyLogger)
    logger = logging.getLogger('Logger')
    logger.setLevel(level)
    # create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)

    # create file handler
    if log_dir is not None and CONFIG.local_rank == 0:
        log_file = os.path.join(log_dir, exp_string)
        fh = logging.FileHandler(log_file+'.log', mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        pprint(CONFIG, stream=fh.stream)

    return logger

