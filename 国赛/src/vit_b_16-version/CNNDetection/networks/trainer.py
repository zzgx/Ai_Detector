import functools
import time
import torch
import torch.nn as nn
from networks.resnet import resnet50 , resnet18, resnet101, resnet152 ,resnet34
from networks.base_model import BaseModel, init_weights
from torchvision import models
import numpy as np
import random

# 设置随机种子
def set_random_seed(seed=2024):
    print(f'Setting random seed {seed}')
    torch.manual_seed(seed)  # 固定CPU上的随机种子
    torch.cuda.manual_seed(seed)  # 固定GPU上的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，固定所有GPU的随机种子

    # 设置cudnn的参数以确保可重复性
    torch.backends.cudnn.deterministic = True  # 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁止自适应地优化，以保证每次运行相同的结果

    # 设置Python的随机种子
    random.seed(seed)  # 固定Python的随机种子

    # 设置NumPy的随机种子
    np.random.seed(seed)  # 固定NumPy的随机种子

def freeze_vit_layers(model):
    # 冻结 conv_proj 和 pos_embedding
    model.conv_proj.weight.requires_grad = False
    model.conv_proj.bias.requires_grad = False
    model.encoder.pos_embedding.requires_grad = False

    # 冻结前6个编码层
    for layer_idx in range(6):
        encoder_layer = getattr(model.encoder.layers, f'encoder_layer_{layer_idx}')
        for param in encoder_layer.parameters():
            param.requires_grad = False

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        seed = int(time.time())  # 获取当前时间戳
        # 固定随机种子
        set_random_seed(seed)

        super(Trainer, self).__init__(opt)

        # print('self.isTrain=',self.isTrain)   # true
        # print('opt.continue_train=', opt.continue_train)  # false

        if self.isTrain and not opt.continue_train:  # 若isTrain为True，且continue_train为False
            # 加载预训练模型
            # self.model = resnet50(pretrained=True)
            self.model = models.vit_b_16(pretrained=True)
            # self.model = models.densenet169(pretrained=True)
            # self.model = models.vgg16(pretrained=True)

            # # 冻结所有层的参数
            # for param in self.model.parameters():
            #     param.requires_grad = False         # True 不冻结

            # # 解冻参数
            # for name, param in self.model.named_parameters():
            #     if "denseblock4" in name:
            #         param.requires_grad = True
            #     if "norm5" in name:
            #         param.requires_grad = True
            #     if "classifier" in name:
            #         param.requires_grad = True

            # # 只解冻某些层，比如最后的几层（layer4 和 fully connected 层）resnet50
            # for param in self.model.layer3.parameters():
            #     param.requires_grad = True
            # for param in self.model.layer4.parameters():
            #     param.requires_grad = True
            # for param in self.model.fc.parameters():
            #     param.requires_grad = True

            # vgg16
            # for name, param in self.model.features.named_parameters():
            #     layer_idx = int(name.split('.')[0])  # 获取层编号
            #     if layer_idx <= 10:  # 冻结前 10 层
            #         param.requires_grad = False
            #     else:  # 解冻后续层
            #         param.requires_grad = True

            # vit_b_16
            freeze_vit_layers(self.model)

            self.model.dropout = nn.Dropout(p=0.5)  # 设置 dropout 层,在forward使用

            # 全连接层
            # self.model.fc = nn.Linear(2048, 2)  # resnet50
            # self.model.classifier[6] = torch.nn.Linear(4096, 2)  # vgg16
            self.model.heads.head = torch.nn.Linear(768, 2)  # vitb16

            # 初始化全连接层权重
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)    # resnet50 初始化fc权重
            # torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, opt.init_gain)  # vgg16 初始化classifier[6]权重
            torch.nn.init.normal_(self.model.heads.head.weight.data, 0.0, opt.init_gain)

        # 损失函数和优化器
        if self.isTrain:                            # 若isTrain为True
            # self.loss_fn = nn.BCEWithLogitsLoss()   # 定义损失函数，BCEWithLogitsLoss是二元交叉熵损失（Binary Cross-Entropy Loss）与 logits 结合的损失函数，BCEWithLogitsLoss 在计算损失时，内部自动应用 sigmoid 函数，将 logits 转换为概率
            self.loss_fn = nn.CrossEntropyLoss()    # 计算交叉熵损失，用于比较两个概率分布之间差异的度量

            # initialize optimizers
            if opt.optim == 'adam':
                # self.optimizer = torch.optim.Adam(self.model.parameters(),
                #                                   lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad,  # 过滤出所有 requires_grad=True 的参数，只优化这些未冻结的参数
                    self.model.parameters()),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),  # betas是Adam优化器中一阶和二阶动量的衰减率
                    weight_decay=1e-4,  # 添加权重衰减，weight_decay 在优化过程中会对模型的权重施加 L2 正则化，防止模型的过拟合
                )

            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        # 打印冻结参数情况
        for name, param in self.model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        self.model.to(opt.gpu_ids[0])

        print(self.model)  # 打印输出模型结构

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        # self.label = input[1].to(self.device).float()   # 标签
        self.label = input[1].to(self.device).long()  # 改为long型标签（整型）

    def forward(self):
        x = self.model(self.input)
        # print(f'output.size(): {x.size()}')  # 打印模型输出维度（batch_size,2）
        x = self.model.dropout(x)  # 将 Dropout 应用于网络的输出
        self.output = x
        return self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output, self.label)
        # 计算预测标签
        # _, predicted_labels = torch.max(self.output, dim=1)  # 取出最大logit的索引作为预测类别
        # predicted_labels = predicted_labels.long()  # 转换为整型标签
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        self.optimizer.step()