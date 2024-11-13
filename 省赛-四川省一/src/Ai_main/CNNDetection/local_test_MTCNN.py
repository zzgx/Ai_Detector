import sys
import time
import os
import csv
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision import datasets
from facenet_pytorch import MTCNN  # 引入 MTCNN
from torch.utils.data import Dataset, DataLoader

# CUDA_VISIBLE_DEVICES=0 python CNNDetection/local_test_MTCNN.py --model_path CNNDetection/model/tag_testdata_B_3_epoch_14_acc_100.0.pth --dataroot tag_testdata_B_3

opt = TestOptions().parse(print_options=False)  # 测试参数：test_options.py
print(f'opt.dataroot = {opt.dataroot}')
print(f'Model_path = {opt.model_path}')

# 创建模型，加载训练好的参数
model = resnet50(num_classes=2)  # nn.Linear(2048, 1)对应num_classes=1
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))  # map_location将模型的参数加载到 CPU 上
model.cuda()
model.eval()
# print(model.training)

# 初始化 MTCNN（用于人脸检测）
mtcnn = MTCNN(image_size=256, margin=30, min_face_size=30, device='cuda' if torch.cuda.is_available() else 'cpu')

# 预处理,与训练时的预处理相同，否则预测结果不正确
transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),  # 居中裁剪到 224x224
                # transforms.RandomRotation(5),  # 随机旋转
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度、对比度
                # transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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
        else:
            print(f"No face detected in {os.path.basename(path)}, using original image.")
            face = img  # 使用原始图像

        # 对裁剪后的人脸或原始图像进行预处理
        if self.transform:
            face = self.transform(face)

        return face, label

# 使用自定义的 ImageFolderWithPaths 加载数据集
test_dataset = FaceDetectionDataset(root=opt.dataroot, transform=transform)

# 使用 ImageFolder 加载数据集，并应用预处理,每个子文件夹的名称作为类标签
# test_dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)

# 使用 DataLoader 加载测试集并批量化
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 准备记录测试结果的变量
correct = 0  # 记录预测正确的样本数
total = 0  # 记录测试集的总样本数

# 结果保存路径
output_csv = "test_on_our_dataset.csv"

# 准备存储图片名称和预测结果的列表
results = []

# 禁用梯度计算
with torch.no_grad():
    for images, labels in test_loader:

        # 将数据迁移到设备（GPU 或 CPU）
        images, labels = images.cuda(), labels.cuda()

        # 模型输出为 logits，我们直接使用 argmax 获取最大概率对应的标签
        output = model(images)  # 输出大小 [batch_size, 2]


        # 无阈值
        # predicted_labels = torch.argmax(output, dim=1)  # 选择最大值的索引作为预测标签

        # 自定义阈值
        # 使用 softmax 计算概率
        probs = torch.nn.functional.softmax(output, dim=1)  # probs 形状为 [batch_size, 2]
        # 获取fake的概率
        fake_probs = probs[:, 1]
        # 定义一个自定义的阈值
        threshold = 0.5
        # 根据阈值判断类别
        predicted_labels = (fake_probs >= threshold).long()  # 如果假类概率 >= 阈值，则预测为假类

        # # 通过 sigmoid 函数将 logits 转化为概率，然后根据概率阈值（通常为 0.5）来做出分类决策。
        # output = model(images).sigmoid()
        # # 使用阈值将输出转换为标签
        # threshold = 0.5    # 阈值可以适当增加或减少
        # predicted_labels = (output >= threshold).float()  # 如果大于或等于threshold，那么这个元素对应的结果为True，否则为 False；float将True转换为 1.0，而 False 会被转换为 0.0。
        # predicted_labels = predicted_labels.long().squeeze()  # 转换为整型标签（0 和 1），squeeze将predicted_label转换为一维张量

        # # print('output=',output)
        # print('images',images)
        # print('predicted_label=', predicted_labels)
        # print('true_label=', labels)

        # 打印预测结果
        batch_size = labels.size(0)
        for j in range(batch_size):
            print(f'Predicted Label: {predicted_labels[j].item()}, True Label: {labels[j].item()}')
            # # 显示图片
            # if j == 5:
            #     img = images[j].cpu()  # 转移到 CPU 以便处理
            #     label = labels[j].item()
            #
            #     # 将张量转换为可显示的图片格式
            #     img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            #     img = (img * 0.229 + 0.485).clip(0, 1)  # 反归一化（根据训练时的均值和标准差）
            #
            #     # 或者直接显示图片（可选）
            #     plt.imshow(img)
            #     plt.title(f"True: {label}  pre: {label}")
            #     plt.axis('off')  # 隐藏坐标轴
            #     plt.show()

        # 统计预测正确的样本数
        total += labels.size(0)  # batch_sized大小
        correct += (predicted_labels == labels).sum().item()
        print('current_batch_acc=', (predicted_labels == labels).sum().item()/labels.size(0)*100)

# 计算准确率
accuracy = correct / total * 100
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')



