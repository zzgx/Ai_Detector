import os
import sys
import time
import torch
import torch.nn
import torch.nn as nn
import argparse
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
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
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练参数
    print('opt.dataroot=',opt.dataroot)
    print(f'opt.validroot:{opt.validroot}')
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))  # 创建了一个日志记录器，所有日志信息会被写入到log.log文件中
    print('  '.join(list(sys.argv)))  # 将程序的命令行参数打印出来

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),  # 居中裁剪到 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度、对比度
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 初始化 MTCNN（用于人脸检测）
    mtcnn = MTCNN(image_size=256, margin=30, min_face_size=30, device='cuda' if torch.cuda.is_available() else 'cpu')

    class FaceDetectionDataset(Dataset):
        def __init__(self, root, transform=None):
            self.dataset = datasets.ImageFolder(root=root)  # 加载数据集
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            # 获取图像及标签
            path, label = self.dataset.imgs[index]
            img = Image.open(path).convert('RGB')  # 转换为 RGB 格式

            # 使用 MTCNN 检测人脸
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                # 如果检测到人脸，裁剪第一个人脸框
                box = boxes[0]
                face = img.crop(box)  # 裁剪人脸

                # # 可选：在原图上绘制检测框
                # draw = ImageDraw.Draw(img)
                # draw.rectangle(box.tolist(), outline="red", width=3)  # 绘制矩形框
                #
                # # 显示裁剪出的人脸
                # plt.imshow(face)
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()

            else:
                print(f"No face detected in {os.path.basename(path)}, using original image.")
                face = img  # 使用原始图像
                # return None, None  # 返回 None 表示这条数据需要被跳过

            # 对裁剪后的人脸进行预处理
            face = self.transform(face)
            return face, label

    # 获取训练数据集加载器
    train_dataset = FaceDetectionDataset(root=opt.dataroot, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dataset_size = len(data_loader)  # 整个训练数据集分成多少个批次
    print('train_batchs = %d' % dataset_size)


    # 使用 ImageFolder 加载数据集，并应用预处理,每个子文件夹的名称作为类标签,root是验证集目录
    val_dataset = FaceDetectionDataset(root=opt.validroot, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    # 在训练过程中将相关的指标（如损失、精度等）记录下来，并在 TensorBoard 中可视化。
    # 可以在训练完成后通过命令 tensorboard --logdir=CNNDetection/checkpoints/4class-resnet-car-cat-chair-horse2024_09_24_12_29_59/train 进行可视化。
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))

    model = Trainer(opt)  # 创建模型

    # 损失不收敛时及时结束训练
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
    class EarlyStopping_acc:
        def __init__(self, patience=3, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_accuracy = 0.0
            self.early_stop = False

        def __call__(self, val_accuracy):
            if val_accuracy > self.best_accuracy + self.min_delta:
                self.best_accuracy = val_accuracy
                self.counter = 0
            else:
                self.counter += 1
                print(f"EarlyStopping_acc counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    # early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    early_stopping = EarlyStopping_acc(patience=10, min_delta=0.01)

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

            # for j in range(len(labels)):
            #     if j == 5:
            #         img = images[j].cpu()  # 转移到 CPU 以便处理
            #         label = labels[j].item()
            #
            #         # 将张量转换为可显示的图片格式
            #         img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            #         img = (img * 0.229 + 0.485).clip(0, 1)  # 反归一化（根据训练时的均值和标准差）
            #
            #         # 直接显示图片
            #         plt.imshow(img)
            #         plt.title(f"True: {label}")
            #         plt.axis('off')  # 隐藏坐标轴
            #         plt.show()


            model.total_steps += 1  # step 表示当前批次数：每循环一个批次自增1
            epoch_iter += opt.batch_size  # 每循环一个批次自增batch_size=16
            # print('model.total_steps=',model.total_steps)
            # print('epoch_iter=', epoch_iter)

            model.set_input(data)  # 输入数据
            model.optimize_parameters()  # 优化模型参数

            if model.total_steps % opt.loss_freq == 0:  # 决定多久显示一次loss
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {} lr {} epoch {}".format(format(model.loss, '.15f'), model.total_steps,
                                                                         model.lr, epoch + 1))  # format,不使用科学计数法
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

                # early stopping
                # loss = criterion(output, val_labels)
                # val_loss += loss.item()

                # 使用 torch.max 选择最大 logit 的索引作为预测类别
                _, predicted_labels = torch.max(output, dim=1)

                # # 使用阈值将输出转换为标签
                # threshold = 0.5  # 阈值可以适当增加或减少
                # predicted_labels = (output.sigmoid() >= threshold).float()  # 如果大于或等于threshold，那么这个元素对应的结果为True，否则为 False；float将True转换为 1.0，而 False 会被转换为 0.0。
                # predicted_labels = predicted_labels.long().squeeze()  # 转换为整型标签（0 和 1），squeeze将predicted_label转换为一维张量


                # 打印预测结果
                batch_size = val_labels.size(0)
                # for j in range(batch_size):
                #     print(f' Predicted Label: {predicted_labels[j].item()}, True Label: {val_labels[j].item()}')
                #     if j == 5 :
                #         img = val_images[j].cpu()  # 转移到 CPU 以便处理
                #         label = val_labels[j].item()
                #         pred_label = predicted_labels[j].item()
                #
                #         # 将张量转换为可显示的图片格式
                #         img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                #         img = (img * 0.229 + 0.485).clip(0, 1)  # 反归一化（根据训练时的均值和标准差）
                #
                #         # 直接显示图片
                #         plt.imshow(img)
                #         plt.title(f"True: {label}, Pred: {pred_label}")
                #         plt.axis('off')  # 隐藏坐标轴
                #         plt.show()

                # 统计预测正确的样本数
                total += val_labels.size(0)  # batch_sized大小
                correct += (predicted_labels == val_labels).sum().item()
                print('current_batch_acc=', (predicted_labels == val_labels).sum().item() / val_labels.size(0) * 100)

            # 计算准确率
            accuracy = correct / total * 100
            print(f'Val Accuracy: {accuracy:.2f}%')
            model.save_networks(f"{current_epoch}_acc_{accuracy}")
            # 检查是否触发 early stopping acc
            print(f"Epoch {epoch + 1}, acc: {accuracy}")
            early_stopping(accuracy)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break



            # # 检查是否触发 early stopping
            # val_loss /= len(val_loader)
            # print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")
            # early_stopping(val_loss)
            # if early_stopping.early_stop:
            #     print("Early stopping triggered")
            #     break

        model.train()

        if epoch % opt.delr_freq == 0 and epoch != 0:  # 改变学习率lr：delr_freq默认为10，即每10个epoch执行一次
            print('epoch=', epoch)
            print('model.total_steps', model.total_steps)
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()

    # 保存模型
    # model.save_networks(current_epoch)