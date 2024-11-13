import os
import subprocess

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 需要执行的命令参数
# model_path：模型存放路径
# dataroot：测试文件夹路径

command = [
    "python", "my_test.py",
    "--model_path", "model/tag_testdata_B_4_epoch_9_acc_99.98333333333333.pth",
    "--dataroot", "../testdata_B"
]

# 执行命令
subprocess.run(command)
