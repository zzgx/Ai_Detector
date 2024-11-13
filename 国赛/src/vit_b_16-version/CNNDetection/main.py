import os
import subprocess

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 需要执行的命令参数
# model_path：模型存放路径
# dataroot：测试文件夹路径

command = [
    "python", "my_test_ensemble.py",
    "--model_path", "model/tag_testdata_B_C1_epoch_10_acc_100.0.pth",
    "--model_path2", "model/B_C1_plan1_epoch_8_acc_97.75.pth",
    "--model_path3", "model/B_C1_plan2_epoch_10_acc_97.75.pth",
    "--model_path4", "model/B_C1_plan3_epoch_10_acc_98.5.pth",
    "--dataroot", "../stable-diffusion-face-dataset-3k_notag"
]

# 执行命令
subprocess.run(command)
