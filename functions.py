# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from utils.utils import make_grid, save_image
from tqdm import tqdm
import cv2

# from utils.fid_score import calculate_fid_given_paths
from utils.torch_fid_score import get_fid
# from utils.inception_score import get_inception_scorepython exps/dist1_new_church256.py --node 0022 --rank 0sample

logger = logging.getLogger(__name__)

def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        # if search_iter < self.grow_step1:
        #     return 0
        # elif self.grow_step1 <= search_iter < self.grow_step2:
        #     return 1
        # else:
        #     return 2
        # for idx, grow_step in enumerate(args.grow_steps):
        #     if iter < grow_step:
        #         return idx
        # return len(args.grow_steps)
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z, epoch).detach()
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_validity = dis_net(fake_imgs)

        # cal loss
        if args.loss == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
            real_label = torch.full((imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss = d_loss/float(args.accumulated_times)
        d_loss.backward()
        
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()

            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % (args.n_critic * args.accumulated_times) == 0:
            
            for accumulated_idx in range(args.g_accumulated_times):
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
                gen_imgs = gen_net(gen_z, epoch)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                loss_lz = torch.tensor(0)
                if args.loss == "standard":
                    real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
                if args.loss == "lsgan":
                    if isinstance(fake_validity, list):
                        g_loss = 0
                        for fake_validity_item in fake_validity:
                            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                            g_loss += nn.MSELoss()(fake_validity_item, real_label)
                    else:
                        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                        # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                        g_loss = nn.MSELoss()(fake_validity, real_label)
                elif args.loss == 'wgangp-mode':
                    fake_image1, fake_image2 = gen_imgs[:args.gen_batch_size//2], gen_imgs[args.gen_batch_size//2:]
                    z_random1, z_random2 = gen_z[:args.gen_batch_size//2], gen_z[args.gen_batch_size//2:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                    torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)

                    g_loss = -torch.mean(fake_validity) + loss_lz
                else:
                    g_loss = -torch.mean(fake_validity)
                g_loss = g_loss/float(args.g_accumulated_times)
                g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            ema_nimg = args.ema_kimg * 1000
            cur_nimg = args.dis_batch_size * args.world_size * global_steps
            if args.ema_warmup != 0:
                ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
                ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
            else:
                ema_beta = args.ema
                
            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                cpu_p = deepcopy(p)
                avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
                del cpu_p

            writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            sample_imgs = torch.cat((gen_imgs[:16], real_imgs[:16]), dim=0)
#             scale_factor = args.img_size // int(sample_imgs.size(3))
#             sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
#             img_grid = make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
#             save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=4, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), ema_beta))
            del gen_imgs
            del real_imgs
            del fake_validity
            del real_validity
            del g_loss
            del d_loss

        writer_dict['train_global_steps'] = global_steps + 1 





def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('calculate Inception score...')
    mean, std = get_inception_score(img_list)

    return mean


def validate(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net.eval()

#     generate images
#     with torch.no_grad():
#         sample_imgs = gen_net(fixed_z, epoch)
#     img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

#     get fid and inception score
#     if args.gpu == 0:
#         fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
#         os.makedirs(fid_buffer_dir, exist_ok=True) if args.gpu == 0 else 0

#     eval_iter = args.num_eval_imgs // args.eval_batch_size
#     img_list = list()
#     for iter_idx in tqdm(range(eval_iter), desc='sample images'):
#         z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    
#         # Generate a batch of images
#         gen_imgs = gen_net(z, epoch).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
#                                                                                                 torch.uint8).numpy()
#         for img_idx, img in enumerate(gen_imgs):
#             file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
#             imsave(file_name, img)
#         img_list.extend(list(gen_imgs))

#     get inception score
    logger.info('=> calculate inception score') if args.rank == 0 else 0
    if args.rank == 0:
#         mean, std = get_inception_score(img_list)
        mean, std = 0, 0
    else:
        mean, std = 0, 0
    print(f"Inception score: {mean}") if args.rank == 0 else 0
#     mean, std = 0, 0
    # get fid score
    print('=> calculate fid score') if args.rank == 0 else 0
    if args.rank == 0:
        fid_score = get_fid(args, fid_stat, epoch, gen_net, args.num_eval_imgs, args.gen_batch_size, args.eval_batch_size, writer_dict=writer_dict, cls_idx=None)
    else:
        fid_score = 10000
    # fid_score = 10000
    print(f"FID score: {fid_score}") if args.rank == 0 else 0
    
#     if args.gpu == 0:
#         if clean_dir:
#             os.system('rm -r {}'.format(fid_buffer_dir))
#         else:
#             logger.info(f'=> sampled images are saved to {fid_buffer_dir}')

#     writer.add_image('sampled_images', img_grid, global_steps)
    if args.rank == 0:
        writer.add_scalar('Inception_score/mean', mean, global_steps)
        writer.add_scalar('Inception_score/std', std, global_steps)
        writer.add_scalar('FID_score', fid_score, global_steps)

        writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


def save_samples(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        batch_size = fixed_z.size(0)
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i:(i+1)], epoch)
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(f"./samples/{args.exp_name}", exist_ok=True)
        save_image(sample_imgs, f'./samples/{args.exp_name}/sampled_images_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(args.num_candidate, with_hidden=True, prev_archs=prev_archs,
                                             prev_hiddens=prev_hiddens)
    hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
            p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
