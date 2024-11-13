import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger


# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']   # 生成器
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]   # 0代表无类别可用，1表示获取文件夹内所有类别


def get_val_opt():  # 获取验证参数
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True   # 按顺序加载数据

    return val_opt




if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练参数
    Testdataroot = os.path.join(opt.dataroot, 'test')   # 生成测试集路径
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)   # 生成训练集路径

    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))  # 创建了一个日志记录器，所有日志信息会被写入到log.log文件中

    print('  '.join(list(sys.argv)))  # 将程序的命令行参数打印出来

    # print('opt.dataroot=',opt.dataroot)
    # print("当前工作目录:", os.getcwd())

    val_opt = get_val_opt()  # 获取验证参数
    Testopt = TestOptions().parse(print_options=False)  # 获取测试参数

    data_loader = create_dataloader(opt)  # 获取训练数据集加载器

    dataset_size = len(data_loader)  # 整个训练数据集分成多少个批次
    print('#training images = %d' % dataset_size)

    # 在训练过程中将相关的指标（如损失、精度等）记录下来，并在 TensorBoard 中可视化。
    # 可以在训练完成后通过命令 tensorboard --logdir=CNNDetection/checkpoints/4class-resnet-car-cat-chair-horse2024_09_24_12_29_59/train 进行可视化。
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)    # 创建模型


    def testmodel():    # 模型测试
        print('*' * 25);
        accs = [];  # 存放精确度
        aps = []    # 存放平均精度分数
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) # 打印当前时间
        for v_id, val in enumerate(vals):   # 对生成器列表循环，v_id是索引，val是值
            # 修改测试参数
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']  # 分类列表
            Testopt.no_resize = True  # testing without resizing by default
            Testopt.no_crop = True  # testing without cropping by default
            # 调用validate方法，传入当前模型和测试参数
            acc, ap, _, _, _, _ = validate(model.model, Testopt)

            accs.append(acc);
            aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc * 100, ap * 100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id + 1, 'Mean', np.array(accs).mean() * 100,
                                                          np.array(aps).mean() * 100));
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    # 模型训练
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):  # 训练多少个epoch
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  # 未使用，统计已处理的数据数量

        for i, data in enumerate(data_loader):  # 训练集加载器
            model.total_steps += 1  # 表示当前批次数：每循环一个批次自增1
            epoch_iter += opt.batch_size  # 每循环一个批次自增batch_size=16
            # print('model.total_steps=',model.total_steps)
            # print('epoch_iter=', epoch_iter)

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:  # 决定多久显示一次loss：loss_freq默认为400，即每隔400steps显示一次
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)  # 记录模型训练过程中的损失值（loss）

        if epoch % opt.delr_freq == 0 and epoch != 0:  # 改变学习率lr：delr_freq默认为10，即每10个epoch执行一次
            print('epoch=',epoch)
            print('model.total_steps',model.total_steps)
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %(epoch, model.total_steps))
            model.adjust_learning_rate()
            model.eval();testmodel();model.train()  # 改变学习率后进行一次模型测试


        # Validation 模型验证
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]    # 验证集验证模型，[:2]取前两个返回值
        val_writer.add_scalar('accuracy', acc, model.total_steps)   # 记录精确度和平均精度分数
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))  # ap：平均精度分数
        model.train()

    model.eval();testmodel()  # 训练完成后（结束所有epoch后）进行一次模型测试

    # 保存模型
    model.save_networks('last')