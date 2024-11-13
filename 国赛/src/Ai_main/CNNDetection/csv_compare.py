import csv


# 读取 CSV 文件并将内容存储为字典，键是文件名，值是标签
def read_csv_to_dict(csv_file):
    data_dict = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行（如果有）
        for row in reader:
            filename, label = row[0], row[1]
            data_dict[filename] = label
    return data_dict


# 计算预测标签的正确率
def calculate_accuracy(true_csv, pred_csv):
    # 读取真实标签和预测标签
    true_labels = read_csv_to_dict(true_csv)
    pred_labels = read_csv_to_dict(pred_csv)

    correct = 0
    total = 0

    # 遍历真实标签的文件名，比较每个文件的预测标签
    for filename, true_label in true_labels.items():
        if filename in pred_labels:  # 确保预测文件中存在相同的文件
            total += 1
            if true_label == pred_labels[filename]:
                correct += 1

    # 计算准确率
    if total == 0:
        return 0.0  # 避免除零错误
    accuracy = correct / total 
    return accuracy


# 指定真实标签和预测标签的CSV文件路径
true_csv_file = '../true_labels_B.csv'
pred_csv_file = '../cla_pre.csv'

# 计算并打印准确率
accuracy = calculate_accuracy(true_csv_file, pred_csv_file)
print(f'相似率: {accuracy * 100:.3f}%')