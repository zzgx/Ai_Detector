import os
import csv

# 定义文件夹路径
root_dir = '../tag_testdata_B_4'  # 父目录，包含0real和1fake文件夹
real_dir = os.path.join(root_dir, '0real')  # 真实图片文件夹路径
fake_dir = os.path.join(root_dir, '1fake')  # 生成图片文件夹路径
output_csv = '../true_labels_B_4.csv'  # 输出的CSV文件路径

# 获取去掉后缀的文件名
def get_file_name_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


# 将图片名称和标签保存到CSV文件中，按字典顺序
def save_images_to_csv(real_dir, fake_dir, output_csv):

    images = []

    # 处理0real文件夹中的图片，打上0标签
    real_images = [(get_file_name_without_extension(f), 0) for f in os.listdir(real_dir) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.extend(real_images)

    # 处理1fake文件夹中的图片，打上1标签
    fake_images = [(get_file_name_without_extension(f), 1) for f in os.listdir(fake_dir) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.extend(fake_images)

    # 对所有图片按字典顺序排序
    images = sorted(images, key=lambda x: x[0])

    # 写入到CSV文件中
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 写入图片名称和标签
        for filename, label in images:
            writer.writerow([filename, label])


# main 函数
def main():
    # 调用函数保存图片名称和标签到CSV
    save_images_to_csv(real_dir, fake_dir, output_csv)
    print(f"图片名称和标签已按字典顺序保存到 {output_csv}")

# 如果此文件是主程序，则执行 main 函数
if __name__ == "__main__":
    main()