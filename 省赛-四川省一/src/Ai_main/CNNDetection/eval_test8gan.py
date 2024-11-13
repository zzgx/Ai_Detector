import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np


# CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --dataroot  {Test-dir} --model_path {Model-Path}

# 对抗学习生成器
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]   # 0代表无分类

opt = TestOptions().parse(print_options=False)  # 测试参数：test_options.py
model_name = os.path.basename(opt.model_path).replace('.pth', '')   # 返回模型文件的名字（去掉扩展名）

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

# get model，创建模型，加载训练好的参数
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))  # map_location将模型的参数加载到 CPU 上
model.cuda()
model.eval()

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    # print('opt.dataroot=',opt.dataroot)  # ../grad_data/test/{val}
    # print('opt.classes=', opt.classes)  # ['bedroom', 'car', 'cat']
    opt.no_resize = True    # testing without resizing by default
    opt.no_crop = True    # testing without cropping by default
    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, acc*100, ap*100))

print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25)