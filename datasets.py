import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from celeba import CelebA

class ImageDataset(object):
    def __init__(self, args, cur_img_size=None, bs=None):
        bs = args.dis_batch_size if bs == None else bs
        img_size = cur_img_size if args.fade_in > 0 else args.img_size
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'celeba':
            Dt = CelebA
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, transform=transform),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = torch.utils.data.DataLoader(
                Dt(root=args.data_path, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))
