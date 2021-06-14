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

os.system(f"CUDA_VISIBLE_DEVICES=0,1 python train_derived.py \
-gen_bs 128 \
-dis_bs 64 \
--dist-url 'tcp://localhost:14256' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank {args.rank} \
--dataset cifar10_flip \
--bottom_width 8 \
--img_size 32 \
--max_iter 500000 \
--gen_model ViT_custom \
--dis_model ViT_custom \
--latent_norm \
--df_dim 384 \
--d_heads 4 \
--d_depth 7 \
--g_depth 5,4,2 \
--dropout 0 \
--latent_dim 1024 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 8 \
--num_eval_imgs 20000 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--D_downsample pixel \
--arch 1 0 1 1 1 0 0 1 1 1 0 1 0 3 \
--patch_size 4 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--exp_name cifar10_flip_ema_500_0.1_latent_norm_D384_G_542_1024_deeper_fade0_wganeps_vit_8_8_color_diff_cutout_n_critic4_bs_g128_d64_glr1_dlr1_beta0_0.99")
