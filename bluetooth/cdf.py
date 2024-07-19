import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def plot_cdf(errors, label, color, cdf_data):
    # 设置CDF图的范围和分辨率
    min_error = 0.0
    max_error = 5.0
    bin_width = 0.1

    # 创建直方图
    hist, bin_edges = np.histogram(errors, bins=np.arange(min_error, max_error + bin_width, bin_width), density=True)
    
    # 计算CDF
    cdf = np.cumsum(hist) * bin_width
    
    # 将CDF数据添加到传递的cdf_data字典中
    cdf_data[label] = {
        'bin_edges': bin_edges[:-1],
        'cdf': cdf
    }
    
    # 绘制CDF图
    plt.plot(bin_edges[:-1], cdf, label=label, color=color)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# 读取CSV文件并提取label和pred列
def read_csv_and_extract_labels_and_preds(file_path):
    df = pd.read_csv(file_path)
    labels = df['label']
    preds = df['pred']
    return labels, preds

# 根据label和pred画出混淆矩阵
def plot_confusion_matrix(file_path):
    # 读取CSV文件中的数据
    labels, preds = read_csv_and_extract_labels_and_preds(file_path)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(labels, preds)

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # 计算评估指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    # 打印评估指标
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')



if __name__ == '__main__':
    # 创建一个字典来保存CDF数据
    cdf_data = {}

    # error = [0,8,0,4,8,4,16,4,16,16,12,0,16,16,0,0,12,16,16,0,]
    error = [0,0,0,0,0,3,0.6,0,0,0,0,0,0.6,0.6,0,0,0,1.2,0,0.6,0,0,0,0,1.2,0,0,0,0.6,0,0.84,1.2,0.6,0.6,0,0,0,0,0,0]
    label = 'ff'
    color = 'blue'
    plot_cdf(error, label, color, cdf_data)


    
    # 绘制图例
    plt.legend()
    plt.show()
    # 示例使用
    file_path = 'C:\\Users\\HONG\\Desktop\\IMU\\test\\bluetooth\\prediction_results_with_distance.csv'  # CSV文件路径
    plot_confusion_matrix(file_path)
    # 将cdf_data中的数据保存到CSV文件
    with open('cdf_data_12s.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入列标题
        writer.writerow(['Bin Edges'] + list(cdf_data.keys()))
        
        # 获取最长的CDF数组长度
        max_length = max(len(data['cdf']) for data in cdf_data.values())
        
        # 写入数据
        for i in range(max_length):
            row = [cdf_data[label]['bin_edges'][i] if i < len(cdf_data[label]['bin_edges']) else '' for label in cdf_data]
            cdf_row = [cdf_data[label]['cdf'][i] if i < len(cdf_data[label]['cdf']) else '' for label in cdf_data]
            writer.writerow(row + cdf_row)
    
    print('CDF data has been saved to cdf_data.csv')