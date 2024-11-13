import sys
import time
import os
import csv
import torch
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
import cv2  # 引入 OpenCV
from torch.utils.data import Dataset, DataLoader

# CUDA_VISIBLE_DEVICES=0 python CNNDetection/local_test.py --model_path CNNDetection/model/test_epoch_4.pth --dataroot tag_testdata_B

opt = TestOptions().parse(print_options=False)  # 测试参数：test_options.py
print(f'opt.dataroot = {opt.dataroot}')
print(f'Model_path = {opt.model_path}')

# 创建模型，加载训练好的参数
model = resnet50(num_classes=2)  # nn.Linear(2048, 1)对应num_classes=1
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))  # map_location将模型的参数加载到 CPU 上
model.cuda()
model.eval()

# 初始化 Haar Cascades（用于人脸检测）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # 将 PIL 图像转换为 numpy 数组
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 使用 Haar Cascades 检测人脸
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # 可以尝试 1.05 或 1.1
            minNeighbors=8,  # 可以尝试 3, 6, 7
        )

        if len(faces) > 0:
            # 如果检测到人脸，裁剪第一个人脸框
            (x, y, w, h) = faces[0]
            face = img_cv[y:y + h, x:x + w]  # 裁剪人脸
        else:
            print(f"No face detected in {os.path.basename(path)}, using original image.")
            face = img_cv  # 使用原始图像

        # 将裁剪后的人脸或原始图像转换为 PIL 格式
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # 对裁剪后的人脸或原始图像进行预处理
        if self.transform:
            face = self.transform(face)

        return face, label

# 使用自定义的 ImageFolderWithPaths 加载数据集
test_dataset = FaceDetectionDataset(root=opt.dataroot, transform=transform)

# 使用 DataLoader 加载测试集并批量化
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 准备记录测试结果的变量
correct = 0  # 记录预测正确的样本数
total = 0  # 记录测试集的总样本数

# 结果保存路径
output_csv = "test_on_our_dataset.csv"

# 准备存储预测结果的列表
results = []

# 禁用梯度计算
with torch.no_grad():
    for images, labels in test_loader:

        # 将数据迁移到设备（GPU 或 CPU）
        images, labels = images.cuda(), labels.cuda()

        # 模型输出为 logits
        output = model(images)  # 输出大小 [batch_size, 2]

        # 使用 softmax 计算概率
        probs = torch.nn.functional.softmax(output, dim=1)  # probs 形状为 [batch_size, 2]
        # 获取 fake 的概率
        fake_probs = probs[:, 1]
        # 定义一个自定义的阈值
        threshold = 0.5
        # 根据阈值判断类别
        predicted_labels = (fake_probs >= threshold).long()  # 如果假类概率 >= 阈值，则预测为假类

        # 打印预测结果
        batch_size = labels.size(0)
        for j in range(batch_size):
            print(f'Predicted Label: {predicted_labels[j].item()}, True Label: {labels[j].item()}')

        # 统计预测正确的样本数
        total += labels.size(0)  # batch_size 大小
        correct += (predicted_labels == labels).sum().item()
        print('current_batch_acc=', (predicted_labels == labels).sum().item()/labels.size(0)*100)

# 计算准确率
accuracy = correct / total * 100
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')
