import os
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from networks.resnet import resnet50
from options.test_options import TestOptions


opt = TestOptions().parse(print_options=False)
print(f'opt.dataroot= {opt.dataroot}')

# 多个模型路径
model_paths = [
    f"{opt.model_path}",
    f"{opt.model_path2}",
    f"{opt.model_path3}",
    f"{opt.model_path4}"
]
print(model_paths)
# 加载所有模型
models = []
for path in model_paths:
    model = resnet50(num_classes=2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.cuda()
    model.eval()
    models.append(model)

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
results = []
threshold = 0.5  # 设定的阈值

# 允许的图像文件扩展名
valid_extensions = ['.jpeg', '.jpg', '.png']

for filename in sorted(os.listdir(opt.dataroot)):
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    img_path = os.path.join(opt.dataroot, filename)
    filename_no_extension = os.path.splitext(filename)[0]

    # 打开图片并进行预处理
    image = Image.open(img_path).convert('RGB')

    # 使用 MTCNN 检测人脸并裁剪
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        box = boxes[0]
        face = image.crop(box)  # 裁剪人脸区域
    else:
        print(f"No face detected in {filename}, using original image.")
        face = image

    # 进行预处理
    face = transform(face)
    face = face.unsqueeze(0)  # 增加batch维度，形状变为 [1, C, H, W]
    face = face.cuda()

    # 初始化用于累积概率的张量
    accumulated_probs = torch.zeros(1, 2).cuda()

    with torch.no_grad():
        for model in models:
            output = model(face)
            probs = torch.nn.functional.softmax(output, dim=1)
            print(f"probs={probs}")
            accumulated_probs += probs  # 累积概率

        # 计算平均概率
        averaged_probs = accumulated_probs / len(models)
        fake_probs = averaged_probs[0, 1]  # 获取假类的平均概率

        # 根据阈值判断类别
        predicted_label = (fake_probs >= threshold).long().item()

    print(filename_no_extension, predicted_label)
    results.append([filename_no_extension, predicted_label])

# 将结果按字典升序排序，区分大小写
results_sorted = sorted(results, key=lambda x: x[0])

# 写入CSV文件
df = pd.DataFrame(results_sorted)
df.to_csv(output_csv, index=False, header=False)

print(f"预测结果已保存到 {output_csv}")
