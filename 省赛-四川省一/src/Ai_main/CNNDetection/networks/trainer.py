import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50 , resnet18
from networks.base_model import BaseModel, init_weights

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # print('self.isTrain=',self.isTrain)   # true
        # print('opt.continue_train=', opt.continue_train)  # false

        if self.isTrain and not opt.continue_train:  # 若isTrain为True，且continue_train为False
            # 预训练的renet50
            self.model = resnet50(pretrained=True)

            self.model.dropout = nn.Dropout(p=0.5)  # Dropout随机让一部分神经元的输出为零,防止过拟合

            # 冻结所有层的参数
            for param in self.model.parameters():
                param.requires_grad = False         # True 不冻结

            # 只解冻某些层，比如最后的几层（layer4 和 fully connected 层）
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

            self.model.fc = nn.Linear(2048, 2)

            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)    # 初始化权重

        if not self.isTrain or opt.continue_train:  # 若isTrain为False或者continue_train为True
            self.model = resnet50(num_classes=1)

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

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        for name, param in self.model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        self.model.to(opt.gpu_ids[0])
 

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
        self.output = self.model(self.input)
        return self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output, self.label)
        # 计算预测标签
        _, predicted_labels = torch.max(self.output, dim=1)  # 取出最大logit的索引作为预测类别
        predicted_labels = predicted_labels.long()  # 转换为整型标签

        # self.forward()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        # output_print = self.output.sigmoid()
        # threshold = 0.5
        # predicted_labels = (output_print >= threshold).float()  # 如果大于或等于threshold，那么这个元素对应的结果为True，否则为 False；float将True转换为 1.0，而 False 会被转换为 0.0。
        # predicted_labels = predicted_labels.long().squeeze()  # 转换为整型标签（0 和 1），squeeze将predicted_label转换为一维张量

        # print('output         =',self.output)
        # print('predicted_label=', predicted_labels)
        # print('true_label     =', self.label)
        # print('current_batch_acc=', (predicted_labels == self.label).sum().item()/self.label.size(0)*100)

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        self.optimizer.step()

    def get_acc(self):
        self.predicted = torch.round(self.output.squeeze(1).sigmoid())
        total += labels.size(0)