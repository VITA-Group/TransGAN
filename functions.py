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
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import cv2

from utils.fid_score import calculate_fid_given_paths
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
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data_1, self.next_data_2 = next(self.loader)
        except StopIteration:
            self.next_data_1 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data_1 = self.next_data_1.cuda(non_blocking=True)
            self.next_data_2 = self.next_data_2.cuda(non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_1, data_2 = self.next_data_1, self.next_data_2
        self.preload()
        return data_1, data_2
        

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
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z, epoch)
            fake_validity = dis_net(gen_imgs)

            # cal loss
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
            else:
                g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            sample_imgs = gen_imgs[:25]
            # scale_factor = args.img_size // int(sample_imgs.size(3))
            # sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=5, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] alpha: %f" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), gen_net.module.alpha))

        writer_dict['train_global_steps'] = global_steps + 1


def train_controller(args, controller, ctrl_optimizer, gen_net, prev_hiddens, prev_archs, writer_dict):
    logger.info("=> train controller...")
    writer = writer_dict['writer']
    baseline = None

    # train mode
    controller.train()

    # eval mode
    gen_net.eval()

    cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        controller_step = writer_dict['controller_steps']
        archs, selected_log_probs, entropies = controller.sample(args.ctrl_sample_batch, prev_hiddens=prev_hiddens,
                                                                 prev_archs=prev_archs)
        cur_batch_rewards = []
        for arch in archs:
            logger.info(f'arch: {arch}')
            gen_net.set_arch(arch, cur_stage)
            is_score = get_is(args, gen_net, args.rl_num_eval_img)
            logger.info(f'get Inception score of {is_score}')
            cur_batch_rewards.append(is_score)
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = cur_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies  # bs * 1
        if baseline is None:
            baseline = cur_batch_rewards
        else:
            baseline = args.baseline_decay * baseline.detach() + (1 - args.baseline_decay) * cur_batch_rewards
        adv = cur_batch_rewards - baseline

        # policy loss
        loss = -selected_log_probs * adv
        loss = loss.sum()

        # update controller
        ctrl_optimizer.zero_grad()
        loss.backward()
        ctrl_optimizer.step()

        # write
        mean_reward = cur_batch_rewards.mean().item()
        mean_adv = adv.mean().item()
        mean_entropy = entropies.mean().item()
        writer.add_scalar('controller/loss', loss.item(), controller_step)
        writer.add_scalar('controller/reward', mean_reward, controller_step)
        writer.add_scalar('controller/entropy', mean_entropy, controller_step)
        writer.add_scalar('controller/adv', mean_adv, controller_step)

        writer_dict['controller_steps'] = controller_step + 1


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
    # sample_imgs = gen_net(fixed_z, epoch)
    # img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    if "celeba" not in args.dataset:
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

            # Generate a batch of images
            gen_imgs = gen_net(z, epoch).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                    torch.uint8).numpy()
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
                imsave(file_name, img)
            img_list.extend(list(gen_imgs))

        # get inception score
        # torch.cuda.empty_cache()
        logger.info('=> calculate inception score')
        mean, std = get_inception_score(img_list)
    else:
        mean, std = 0, 0
    print(f"Inception score: {mean}")
    # mean, std = 0, 0
    # get fid score
    logger.info('=> calculate fid score')
    fid_score = get_fid(args, fid_stat, epoch, gen_net, args.num_eval_imgs, args.gen_batch_size*2, writer_dict=writer_dict, cls_idx=None)
    # fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    # fid_score = 10000
    print(f"FID score: {fid_score}")

    if clean_dir:
        os.system('rm -r {}'.format(fid_buffer_dir))
    else:
        logger.info(f'=> sampled images are saved to {fid_buffer_dir}')

    # writer.add_image('sampled_images', img_grid, global_steps)
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
