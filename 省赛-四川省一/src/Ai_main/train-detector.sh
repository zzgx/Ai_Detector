#!/bin/bash

pwd=$(cd $(dirname $0); pwd) # 保存当前脚本所在目录的绝对路径
cd ${pwd}/CNNDetection/  # cd CNNDetection
#echo ${pwd}

# train
# img2grad 会使精度下降
# 引入MTCNN人脸识别增加精度
# 因为ResNet50已经在大规模数据集（ImageNet）上预训练，模型主要是微调（Fine-tuning），所以通常 10-20个 Epoch 就足够了
# ./train-detector.sh {Grad-Save-Dir}
# ./train-detector.sh ../tag_testdata_B_4

CUDA_VISIBLE_DEVICES=0 python my_train_MTCNN.py --name my_train_model --dataroot $1 --batch_size 64 --delr_freq 10 --lr 0.0001 --niter 20 --loss_freq 10 --optim adam

# test
# CUDA_VISIBLE_DEVICES=0 python CNNDetection/test_on_our_dataset.py --model_path CNNDetection/model/SD_10k_noGrad_model_epoch_5.pth --dataroot tag_testdata
# CUDA_VISIBLE_DEVICES=0 python CNNDetection/my_test.py --model_path CNNDetection/model/tagTestData_model_epoch_5.pth  --dataroot tag_testdata

# 修改：
# CNNDetection/data/__init__.py -- 8：num_workers=0 多线程导致pickle错误