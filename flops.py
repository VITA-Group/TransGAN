# -*- coding: utf-8 -*-
# @Date    : 2019-10-01
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models_search
import datasets
from functions import train, validate, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from models_search.ViT_8_8 import matmul, count_matmul


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


    # set tf env
    # _init_inception()
    # inception_path = check_or_download_inception(None)
    # create_inception_graph(inception_path)

    # # import network
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args).cuda()
    gen_net.set_arch(args.arch, cur_stage=2)

    import thop, math
    dummy_data = (1, 1024)
    macs, params = thop.profile(gen_net, inputs=(torch.randn(dummy_data).cuda(), ),
                        custom_ops={matmul: count_matmul})
    flops, params = thop.clever_format([macs, params], "%.3f")
    print('Flops (GB):\t', flops)
    print('Params Size (MB):\t', params)


if __name__ == '__main__':
    main()
