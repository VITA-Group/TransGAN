# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import cv2

from utils.fid_score import calculate_fid_given_paths
# from utils.torch_fid_score import get_fid
# from utils.inception_score import get_inception_score

logger = logging.getLogger(__name__)

import cfg
import models_search
from functions import validate
from utils.utils import set_log_dir, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from utils.inception_score import get_inception_score

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def validate(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net.eval()

#     generate images
    with torch.no_grad():
#         sample_imgs = gen_net(fixed_z, epoch)
#         img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)


        eval_iter = args.num_eval_imgs // args.eval_batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

            # Generate a batch of images
            gen_imgs = gen_net(z, epoch).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            img_list.extend(list(gen_imgs))

#     mean, std = 0, 0
    # get fid score
#     mean, std = get_inception_score(img_list)
#     print(f"IS score: {mean}")
    print('=> calculate fid score') if args.rank == 0 else 0
    fid_score = calculate_fid_given_paths([img_list, fid_stat], inception_path=None)
    # fid_score = 10000
    print(f"FID score: {fid_score}") if args.rank == 0 else 0
    with open(f'output/{args.exp_name}.txt', 'a') as f:
        print('fid:' + str(fid_score) + 'epoch' + str(epoch), file=f)
    
    if args.rank == 0:
#         writer.add_scalar('Inception_score/mean', mean, global_steps)
#         writer.add_scalar('Inception_score/std', std, global_steps)
        writer.add_scalar('FID_score', fid_score, global_steps)

#         writer_dict['valid_global_steps'] = global_steps + 1

    return 0, fid_score

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
#     assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=[0])

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'cifar10_flip':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.fid_stat is not None:
        fid_stat = args.fid_stat
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (4, args.latent_dim)))

    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }
    inception_score, fid_score = validate(args, fixed_z, fid_stat, epoch, gen_net, writer_dict, clean_dir=False)
    logger.info(f'Inception score: {inception_score}, FID score: {fid_score}.')


if __name__ == '__main__':
    main()
