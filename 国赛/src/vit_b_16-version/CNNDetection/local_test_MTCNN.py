import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN  # 引入 MTCNN
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from networks.resnet import resnet50
from options.test_options import TestOptions
from torchvision import models

# CUDA_VISIBLE_DEVICES=0 python CNNDetection/local_test_MTCNN.py --model_path CNNDetection/model/vitb16_tag_testdata_B_C1_epoch_7_acc_100.0.pth --dataroot latent_diffusion

opt = TestOptions().parse(print_options=False)  # 测试参数：test_options.py
print(f'opt.dataroot = {opt.dataroot}')
print(f'Model_path = {opt.model_path}')

# 创建模型，加载训练好的参数
# model = resnet50(num_classes=2)

# model = models.vgg16(weights=None)
# model.classifier[6] = torch.nn.Linear(4096, 2)  # 修改vgg16输出维度

model = models.vit_b_16(weights=None)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)  # 确保最后一层为2个输出，二分类

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=False)  # map_location将模型的参数加载到 CPU 上

model.cuda()
model.eval()
print(model)

# 初始化 MTCNN（用于人脸检测）
mtcnn = MTCNN(image_size=256, margin=30, min_face_size=30, device='cuda' if torch.cuda.is_available() else 'cpu')

# 预处理,与训练时的预处理相同，否则预测结果不正确
transform = transforms.Compose([
                transforms.Resize((224, 224)),
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
        # print(f'output.size()={output.size()}')

        # 自定义阈值
        # 使用 softmax 计算概率
        probs = torch.nn.functional.softmax(output, dim=1)  # probs 形状为 [batch_size, 2]
        # print(f'probs={probs}')
        # 获取fake的概率
        fake_probs = probs[:, 1]
        # print(f'fake_probs={fake_probs}')
        # 定义一个自定义的阈值
        threshold = 0.5
        # 根据阈值判断类别
        predicted_labels = (fake_probs >= threshold).long()  # 如果假类概率 >= 阈值，则预测为假类

        # 打印预测结果
        batch_size = labels.size(0)
        for j in range(batch_size):
            print(f'Predicted Label: {predicted_labels[j].item()}, True Label: {labels[j].item()}')

        # 统计预测正确的样本数
        total += labels.size(0)  # batch_sized大小
        correct += (predicted_labels == labels).sum().item()
        print('current_batch_acc=', (predicted_labels == labels).sum().item()/labels.size(0)*100)

# 计算准确率
accuracy = correct / total * 100
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')