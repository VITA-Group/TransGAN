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

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_derived.py \
-gen_bs 16 \
-dis_bs 16 \
--accumulated_times 4 \
--g_accumulated_times 8 \
--dist-url 'tcp://localhost:10641' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank {args.rank} \
--dataset church \
--data_path ./lsun \
--bottom_width 8 \
--img_size 256 \
--max_iter 500000 \
--gen_model ViT_custom_local544444_256_rp \
--dis_model ViT_scale3_local_new_rp \
--g_window_size 16 \
--d_window_size 16 \
--g_norm pn \
--df_dim 384 \
--d_depth 3 \
--g_depth 5,4,4,4,4,4 \
--latent_dim 512 \
--gf_dim 1024 \
--num_workers 0 \
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
--val_freq 5000 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--D_downsample pixel \
--arch 1 0 1 1 1 0 0 1 1 1 0 1 0 3 \
--patch_size 4 \
--diff_aug translation,erase_ratio,color,hue \
--fid_stat fid_stat/fid_stats_church_256.npz \
--ema 0.995 \
--exp_name church_256")



