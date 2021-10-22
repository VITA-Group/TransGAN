#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
-gen_bs 32 \
-dis_bs 16 \
--accumulated_times 4 \
--g_accumulated_times 4 \
--dist-url 'tcp://localhost:10641' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank {args.rank} \
--dataset celeba \
--data_path ./celeba_hq \
--bottom_width 8 \
--img_size 256 \
--max_iter 500000 \
--gen_model Celeba256_gen \
--dis_model Celeba256_dis \
--g_window_size 16 \
--d_window_size 4 \
--g_norm pn \
--df_dim 384 \
--d_depth 3 \
--g_depth 5,4,4,4,4,4 \
--latent_dim 512 \
--gf_dim 1024 \
--num_workers 32 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 10 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 10 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--diff_aug filter,translation,erase_ratio,color,hue \
--fid_stat fid_stat/fid_stats_celeba_hq_256.npz \
--ema 0.995 \
--load_path ./celeba_256_checkpoint \
--exp_name celeba_hq_256")



