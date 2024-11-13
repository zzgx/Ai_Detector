import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

# 测试（验证）过程
def validate(model, opt):

    # 根据传入参数返回对应的数据集加载器
    data_loader = create_dataloader(opt)

    with torch.no_grad():  # 禁用梯度计算
        # y_true 用于存储真实标签，y_pred 用于存储模型预测的标签。
        y_true, y_pred = [], []
        for img, label in data_loader:  # 遍历data_loader中的每个批次
            in_tens = img.cuda()  # GPU加速计算

            # 将 in_tens 输入到模型中进行前向传播，得到模型的输出。
            # 使用 sigmoid() 函数将输出转换为概率值（通常用于二分类问题）。
            # 使用 flatten() 将多维输出展平为一维，然后转换为列表并添加到 y_pred 中。
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())

            # 将真实标签 label 展平为一维数组，转换为列表并添加到 y_true 中。
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)  # 将 y_true 和 y_pred 转换为 NumPy 数组，方便后续的计算
    # 计算模型判断真实图像的准确率,y_true[y_true==0]的y_true==0标识哪些样本的真实标签是0，再通过y_true[y_true==0]取值
    # （y_pred[y_true==0] > 0.5）是找出预测标签里对应y_true==0索引位置的值，如果预测结果大于0.5，则返回True，否则返回False
    # accuracy_score计算它们之间的准确度
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    # 计算模型判断虚假图像的准确率
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)  # 计算整体准确率，accuracy_score只在单一阈值（如 0.5）下计算准确率，容易忽略模型在其他阈值下的表现。
    ap = average_precision_score(y_true, y_pred)    # 计算平均精度，average_precision_score 会较高，因为它考虑了所有阈值。
    return acc, ap, r_acc, f_acc, y_true, y_pred    # 返回 整体准确率 acc、平均精度 ap、真实图像的准确率r_acc、判断虚假图像的准确率f_acc、真实标签列表 y_true 和预测结果列表、


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)
    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)