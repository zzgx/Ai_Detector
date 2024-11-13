import os
import pandas as pd
import shutil

# 根据csv将图像分类

# 配置路径
csv_path = 'D:\AI_competition\Ai_main\cla_pre_B.csv'  # CSV 文件路径
images_folder = 'D:\AI_competition\Ai_main\\testdata_B'  # 图片所在文件夹路径
output_folder = 'D:\AI_competition\Ai_main\\tag_testdata_B'  # 输出文件夹路径

# 创建标签文件夹（如果不存在）
for label in [0, 1]:  # 假设只有0和1两种标签
    label_folder = os.path.join(output_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_path, header=None, names=['image_name', 'label'])

# 处理每一行，将图片复制到对应标签的文件夹
for idx, row in df.iterrows():
    image_name = f"{row['image_name']}.jpg"  # 假设图片格式为 .jpg
    label = str(row['label'])

    # 图片的原路径和目标路径
    src_path = os.path.join(images_folder, image_name)
    dest_path = os.path.join(output_folder, label, image_name)

    # 检查图片是否存在
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")
    else:
        print(f"Image {src_path} not found.")

print("图片分类完成！")
