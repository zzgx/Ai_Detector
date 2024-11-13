from facenet_pytorch import MTCNN
from PIL import Image
import torch
from torchvision import transforms
# 初始化 MTCNN
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# 打开一张图像
img = Image.open('../testdata_B/dCj01nXLD2Qyq4im.jpg')

# 检测并裁剪人脸
face = mtcnn(img)

if face is not None:
    # 将 Tensor 转换为 PIL 图像，便于显示
    face_img = transforms.ToPILImage()(face.cpu().clamp(0, 1))  # CHW -> RGB
    face_img.show()  # 显示裁剪后的人脸
else:
    print("No face detected.")
