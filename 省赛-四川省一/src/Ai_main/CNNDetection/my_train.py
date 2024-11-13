import os
import sys
import time
import torch
import torch.nn
import torch.nn as nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from validate import validate
from data import create_dataloader
from data import create_dataloader_2
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from torchvision import datasets
from torchvision import transforms

if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练参数
    print('opt.dataroot=',opt.dataroot)

    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))  # 创建了一个日志记录器，所有日志信息会被写入到log.log文件中
    print('  '.join(list(sys.argv)))  # 将程序的命令行参数打印出来

    data_loader = create_dataloader_2(opt)  # 获取训练数据集加载器（改）
    dataset_size = len(data_loader)  # 整个训练数据集分成多少个批次
    print('batchs = %d' % dataset_size)

    # 预处理,必须与训练时的预处理相同，否则预测结果不正确
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度、对比度
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 使用 ImageFolder 加载数据集，并应用预处理,每个子文件夹的名称作为类标签,root是验证集目录
    val_dataset = datasets.ImageFolder(root='../tag_testdata_B', transform=transform)

    # 使用 DataLoader 加载验证集并批量化
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # 在训练过程中将相关的指标（如损失、精度等）记录下来，并在 TensorBoard 中可视化。
    # 可以在训练完成后通过命令 tensorboard --logdir=CNNDetection/checkpoints/4class-resnet-car-cat-chair-horse2024_09_24_12_29_59/train 进行可视化。
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    model = Trainer(opt)  # 创建模型

    class EarlyStopping:
        def __init__(self, patience=2, min_delta=0.0):
            """
            Args:
                patience (int): 如果验证损失在 `patience` 个 epoch 内没有改善，则停止训练。
                min_delta (float): 验证损失的最小改善幅度。
            """
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0  # 不提升次数计数
            self.best_loss = float('inf')
            self.early_stop = False

        def __call__(self, val_loss):
            """
            检查验证损失是否有提升，如果没有，增加计数器。
            """
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0  # 重置计数器
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    # 模型训练
    model.train()
    current_epoch = 0
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):  # 训练多少个epoch
        current_epoch += 1
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  # 统计已处理的数据数量

        for i, data in enumerate(data_loader):  # 训练集加载器
            images, labels = data

            model.total_steps += 1  # step 表示当前批次数：每循环一个批次自增1
            epoch_iter += opt.batch_size  # 每循环一个批次自增batch_size=16
            # print('model.total_steps=',model.total_steps)
            # print('epoch_iter=', epoch_iter)

            model.set_input(data)   # 输入数据
            model.optimize_parameters()  # 优化模型参数

            if model.total_steps % opt.loss_freq == 0:  # 决定多久显示一次loss
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {} lr {} epoch {}".format(format(model.loss, '.15f'), model.total_steps, model.lr, epoch+1))   # format,不使用科学计数法
                train_writer.add_scalar('loss', model.loss, model.total_steps)  # 记录模型训练过程中的损失值（loss）

        # 模型验证
        model.eval()
        total = 0
        correct = 0
        # 禁用梯度计算
        val_loss = 0.0
        # criterion = nn.BCEWithLogitsLoss()  # 定义验证集损失函数
        criterion = nn.CrossEntropyLoss()  # 改为 CrossEntropyLoss
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_images, val_labels = data

                # 将数据迁移到设备（GPU 或 CPU）
                val_images, val_labels = val_images.cuda(), val_labels.cuda()

                # 通过 sigmoid 函数将 logits 转化为概率，然后根据概率阈值（通常为 0.5）来做出分类决策。
                model.set_input(data)
                output = model.forward()
                loss = criterion(output, val_labels)
                val_loss += loss.item()

                # 使用 torch.max 选择最大 logit 的索引作为预测类别
                _, predicted_labels = torch.max(output, dim=1)

                # # 使用阈值将输出转换为标签
                # threshold = 0.5  # 阈值可以适当增加或减少
                # predicted_labels = (output.sigmoid() >= threshold).float()  # 如果大于或等于threshold，那么这个元素对应的结果为True，否则为 False；float将True转换为 1.0，而 False 会被转换为 0.0。
                # predicted_labels = predicted_labels.long().squeeze()  # 转换为整型标签（0 和 1），squeeze将predicted_label转换为一维张量

                # 打印预测结果
                batch_size = val_labels.size(0)
                for j in range(batch_size):
                    print(f' Predicted Label: {predicted_labels[j].item()}, True Label: {val_labels[j].item()}')

                # 统计预测正确的样本数
                total += val_labels.size(0)  # batch_sized大小
                correct += (predicted_labels == val_labels).sum().item()
                print('current_batch_acc=', (predicted_labels == val_labels).sum().item() / val_labels.size(0) * 100)

            # 计算准确率
            accuracy = correct / total * 100
            print(f'Val Accuracy: {accuracy:.2f}%')

            # 检查是否触发 early stopping
            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        model.train()

        if epoch % opt.delr_freq == 0 and epoch != 0:  # 改变学习率lr：delr_freq默认为10，即每10个epoch执行一次
            print('epoch=', epoch)
            print('model.total_steps', model.total_steps)
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()

    # 保存模型
    model.save_networks(current_epoch)