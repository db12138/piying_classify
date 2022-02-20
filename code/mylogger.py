#文件名 logger_time_epoch_hidden_headsnum_lr_drop
#列 epoch train_acc dev_acc train_loss dev_loss epoch_num hidden headsnum lr drop
#pandas读取
#Python 读取画散点图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from colorlog import ColoredFormatter
import sys
from time import gmtime, strftime
def create_logger(name,log_path,to_disk=True,prefix=None):
    log = logging.getLogger(name)
    level = logging.INFO
    log.setLevel(level)
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s [%(blue)s%(message)s%(reset)s]",
        datefmt='%Y-%m-%d %I:%M:%S',
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    fformatter = logging.Formatter(
        "%(asctime)s [%(funcName)-12s] %(levelname)-8s [%(message)s]",
        datefmt='%Y-%m-%d %I:%M:%S',
        style='%'
    )

    if to_disk:
        prefix = prefix if prefix is not None else 'my_log'
        time_stamp = strftime('{}-%Y-%m-%d-%H-%M-%S.log'.format(prefix), gmtime())
        log_file_path = log_path + time_stamp
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fformatter)
        log.addHandler(fh)
    return log

if __name__ == '__main__':
    fileDIR  = "logs/"
    





