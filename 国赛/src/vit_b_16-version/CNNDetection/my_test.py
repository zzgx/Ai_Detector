import os
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN  # 引入 MTCNN
from torchvision import transforms
from networks.resnet import resnet50
from options.test_options import TestOptions

# CUDA_VISIBLE_DEVICES=0 python CNNDetection/my_test.py --model_path CNNDetection/model/SD-human-7k_epoch_5_B-0-4614.pth --dataroot testdata_B

opt = TestOptions().parse(print_options=False)   # 测试参数：test_options.py
print(f'opt.dataroot= {opt.dataroot}')
print(f'Model_path= {opt.model_path}')

# get model，创建模型，加载训练好的参数
model = resnet50(num_classes=2)

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))  # map_location将模型的参数加载到 CPU 上

# print(model)
model.cuda()
model.eval()

# 初始化 MTCNN（用于人脸检测）
mtcnn = MTCNN(image_size=256, margin=30, min_face_size=30, device='cuda' if torch.cuda.is_available() else 'cpu')

# 预处理
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

# 结果保存路径
output_csv = "../cla_pre.csv"

# 准备存储图片名称和预测结果的列表
results = []

threshold = 0.5  # 阈值可以适当增加或减少

# 允许的图像文件扩展名
valid_extensions = ['.jpeg', '.jpg', '.png']

# csv中0真，1假
for filename in sorted(os.listdir(opt.dataroot)):
    # 过滤非图像文件
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue
    # 获取图片的完整路径
    img_path = os.path.join(opt.dataroot, filename)
    # print('img_path=', img_path)
    # 去掉后缀
    filename_no_extension = os.path.splitext(filename)[0]

    # 打开图片并进行预处理
    image = Image.open(img_path).convert('RGB')
    # 使用 MTCNN 检测人脸并裁剪
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        # 如果检测到人脸，裁剪第一个人脸框
        box = boxes[0]
        face = image.crop((box[0], box[1], box[2], box[3]))  # 裁剪人脸区域
    else:
        print(f"No face detected in {filename}, using original image.")
        face = image  # 如果没有检测到人脸，使用原始图像
    # 进行预处理
    face = transform(face)
    face = face.unsqueeze(0)  # 增加batch维度，形状变为 [1, C, H, W]
    face = face.cuda()

    # print('image=', image)

    # 将图片输入模型并得到预测结果
    with torch.no_grad():

        output = model(face)

        # 无阈值
        # predicted_label = torch.argmax(output, dim=1).item()  # 获取最大值的索引作为预测标签

        # 自定义阈值
        # 使用 softmax 计算概率
        probs = torch.nn.functional.softmax(output, dim=1)  # probs 形状为 [batch_size, 2]
        # 获取fake的概率
        fake_probs = probs[:, 1]
        # 定义一个自定义的阈值
        threshold = 0.5
        # 根据阈值判断类别
        predicted_label = (fake_probs >= threshold).long().item()  # 如果假类概率 >= 阈值，则预测为假类

    print(filename_no_extension, predicted_label)
    # 将图片名称和预测结果添加到列表
    results.append([filename_no_extension, predicted_label])

# 按字典升序排序，区分大小写
results_sorted = sorted(results, key=lambda x: x[0])

# 将结果写入CSV文件
df = pd.DataFrame(results_sorted)
df.to_csv(output_csv, index=False, header=False)

print(f"预测结果已保存到 {output_csv}")