#!/bin/bash

pwd=$(cd $(dirname $0); pwd) # 保存当前脚本所在目录的绝对路径
cd ${pwd}/CNNDetection/  # cd CNNDetection
#echo ${pwd}

# train
# img2grad 会使精度下降
# 引入MTCNN人脸识别增加精度
# 因为模型已经在大规模数据集（ImageNet）上预训练，模型主要是微调（Fine-tuning），所以通常 10-20个 Epoch 就足够了
# ./train-detector.sh {Save-Dir}
# ./train-detector.sh ../B_C1_plan3 ../valide_B_C1_plan3

CUDA_VISIBLE_DEVICES=0 python my_train_MTCNN.py --name my_train_model --dataroot $1 --validroot $2 --batch_size 64 --delr_freq 10 --lr 0.0001 --niter 10 --loss_freq 10 --optim adam

# 修改：
# CNNDetection/data/__init__.py -- 8：num_workers=0 多线程导致pickle错误