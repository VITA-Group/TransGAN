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
from torch.autograd import Variable
from imageio import imsave
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import cv2
from torch.autograd import grad

# from utils.fid_score import calculate_fid_given_paths
from utils.torch_fid_score import get_fid
from utils.inception_score import get_inception_score

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
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = Variable(interpolates, requires_grad = True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty



def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    dis_net.module.cur_stage = gen_net.module.cur_stage
    

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        if gen_net.module.alpha < 1:
            gen_net.module.alpha += args.fade_in
            gen_net.module.alpha = min(1., gen_net.module.alpha)
            dis_net.module.alpha = gen_net.module.alpha

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).to("cuda:0")

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).to(real_imgs.get_device())

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs, _ = gen_net(z, epoch)
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_validity = dis_net(fake_imgs.detach())

        # cal loss
        if args.loss == 'hinge':
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'style':
            real_loss = torch.nn.functional.softplus(-real_validity)
            fake_loss = torch.nn.functional.softplus(-fake_validity)
            d_loss = real_loss.mean() + fake_loss.mean()

        elif args.loss == 'standard':
            real_label = torch.full((imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'ragan':
            BCE_stable = torch.nn.BCEWithLogitsLoss()
            real_label = torch.full((imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
            real_validity = real_validity.view(-1)
            fake_validity = fake_validity.view(-1)
            d_loss = (BCE_stable(real_validity - torch.mean(fake_validity), real_label) + 
                    BCE_stable(fake_validity - torch.mean(real_validity), fake_label))/2
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        elif args.loss == 'wgangp-eps-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 1.)
        dis_optimizer.step()

        args.d_reg_every = 8
        args.r1 = 10
        d_regularize = iter_idx % args.d_reg_every == 0
        if d_regularize and args.loss == "style":
            real_imgs.requires_grad = True
            real_pred = dis_net(real_imgs)
            r1_loss = d_r1_loss(real_pred, real_imgs)

            dis_net.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            dis_optimizer.step()

        #writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs, sr_img = gen_net(gen_z, epoch, real_imgs)

            fake_validity = dis_net(gen_imgs)

            # cal loss
            loss_lz = torch.tensor(0)
            if args.loss == "standard":
                real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
            elif args.loss == 'ragan':
                real_validity = dis_net(real_imgs)
                real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
                real_validity = real_validity.view(-1)
                fake_validity = fake_validity.view(-1)
                g_loss = (BCE_stable(real_validity - torch.mean(fake_validity), fake_label) + 
                            BCE_stable(fake_validity - torch.mean(real_validity), real_label))/2
            elif args.loss == 'style':
                g_loss = torch.nn.functional.softplus(-fake_validity).mean()
            elif args.loss == 'wgangp-eps-mode':
                fake_image1, fake_image2 = gen_imgs[:args.gen_batch_size//2], gen_imgs[args.gen_batch_size//2:]
                z_random1, z_random2 = gen_z[:args.gen_batch_size//2], gen_z[args.gen_batch_size//2:]
                lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                torch.abs(z_random2 - z_random1))
                eps = 1 * 1e-5
                loss_lz = 1 / (lz + eps)

                g_loss = -torch.mean(fake_validity) + loss_lz
            else:
                g_loss = -torch.mean(fake_validity)
            mse_loss = nn.MSELoss()(sr_img, real_imgs)
            mse_weight = 50 if epoch<110 else 0
            g_loss += mse_loss*mse_weight
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 1.)
            gen_optimizer.step()

            # g_regularize = i % args.g_reg_every == 0
            # if g_regularize and args.loss == "style":


            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                #writer.add_scalar('LR/g_lr', g_lr, global_steps)
                #writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            #writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            sample_imgs, _ = gen_net(fixed_z, epoch)
            scale_factor = args.img_size // int(sample_imgs.size(3))
            sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
            img_grid = make_grid(sample_imgs, nrow=8, normalize=True, scale_each=True)
            save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=8, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [MSE loss: %f] [Mode loss: %f] alpha: %f" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), mse_loss.item(), loss_lz.item(), gen_net.module.alpha))

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
    gen_net = gen_net.eval()

    # generate images
    sample_imgs, _ = gen_net(fixed_z, epoch)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

#         Generate a batch of images
        gen_imgs = gen_net(z, epoch)[0].mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

#     get inception score
#     torch.cuda.empty_cache()
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)
    # mean, std = 0, 0
    print(f"Inception score: {mean}")

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = get_fid(args, fid_stat, epoch, gen_net, args.num_eval_imgs, args.gen_batch_size*2, writer_dict=writer_dict, cls_idx=None)
    # fid_score = 10000
    print(f"FID score: {fid_score}")

    if clean_dir:
        os.system('rm -r {}'.format(fid_buffer_dir))
    else:
        logger.info(f'=> sampled images are saved to {fid_buffer_dir}')
	
    #writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


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


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

