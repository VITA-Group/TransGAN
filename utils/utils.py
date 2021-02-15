# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz
import torch


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))
