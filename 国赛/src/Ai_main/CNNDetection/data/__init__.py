import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder


def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes: # 按类别进行循环
        root = opt.dataroot + '/' + cls
        # root = opt.dataroot + cls
        # print('root=', root)
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)  # ConcatDataset 是 PyTorch 的一个工具，可以将多个数据集拼接在一起，像一个单独的数据集一样使用。

def get_dataset_2(opt):
    dset_lst = []
    root = opt.dataroot
    # root = opt.dataroot + cls
    # print('root=', root)
    dset = dataset_folder(opt, root)
    dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)  # ConcatDataset 是 PyTorch 的一个工具，可以将多个数据集拼接在一起，像一个单独的数据集一样使用。

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    # 当处于训练模式（opt.isTrain 为 True）且（opt.class_bal 为 False）时，shuffle 的值由 not opt.serial_batches 决定，即如果不按顺序加载数据（opt.serial_batches 为 False），则 shuffle 为 True，表示打乱数据。
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)  # 调用get_dataset方法
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    # print('opt.num_threads=',opt.num_threads)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              # num_workers=int(opt.num_threads)  # 多线程导致pickle错误
                                              num_workers=0
                                              )
    return data_loader

def create_dataloader_2(opt):
    # 当处于训练模式（opt.isTrain 为 True）且（opt.class_bal 为 False）时，shuffle 的值由 not opt.serial_batches 决定，即如果不按顺序加载数据（opt.serial_batches 为 False），则 shuffle 为 True，表示打乱数据。
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset_2(opt)  # 调用get_dataset方法
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    # print('opt.num_threads=',opt.num_threads)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              # shuffle=shuffle,
                                              shuffle=True,
                                              sampler=sampler,
                                              # num_workers=int(opt.num_threads)  # 多线程导致pickle错误
                                              num_workers=0
                                              )
    return data_loader