import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import math
def map_number_to_coordinate(number):
    coordinates = {
        1: (0, 0),
        2: (0, 0.6),
        3: (0, 1.2),
        4: (0, 1.8),
        5: (0, 2.4),
        6: (0, 3.0),
        7: (0, 3.6),
        8: (0, 4.2),
        9: (0, 4.8),
        10: (0, 5.4),
        11: (3, 0),
        12: (3, 0.6),
        13: (3, 1.2),
        14: (3, 1.8),
        15: (3, 2.4),
        16: (3, 3.0),
        17: (3, 3.6),
        18: (3, 4.2),
        19: (3, 4.8),
        20: (3, 5.4),
        21: (6, 0),
        22: (6, 0.6),
        23: (6, 1.2),
        24: (6, 1.8),
        25: (6, 2.4),
        26: (6, 3.0),
        27: (6, 3.6),
        28: (6, 4.2),
        29: (6, 4.8),
        30: (6, 5.4),
        31: (0, 6),
        32: (0.6, 6),
        33: (1.2, 6),
        34: (1.8, 6),
        35: (2.4, 6),
        36: (3.0, 6),
        37: (3.6, 6),
        38: (4.2, 6),
        39: (4.8, 6),
        40: (5.4, 6),
        41: (6, 6)
    }
    x, y = coordinates.get(number, (0, 0))  # 默认值为 (0, 0)
    return Point(x, y)

# 定义 Point 类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 计算两个点之间的距离
def calculate_distance(p1, p2):
    distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    return distance

# 加载训练好的模型
model = load_model('C:\\Users\\HONG\\Desktop\\IMU\\test\\bluetooth\\Experiment-main\\model_comparison\\0717_with\\DNN\\0717_with.h5')

# 从 CSV 文件中读取数据到 DataFrame
data = pd.read_csv('C:\\Users\\HONG\\Desktop\\IMU\\test\\0717_testing_7.csv')

# 假设数据的列名为 'rssi1', 'rssi2', ..., 'rssi7'，并且最后一列为目标变量（one-hot编码）
# 假设您的数据中有七个beacon的RSSI值，可以通过以下代码选择这些列作为输入数据
input_data = data[['Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7']]
labels = data['label']  # 选择 'label' 列作为标签列

# 将输入数据转换为 numpy 数组
input_data_array = input_data.to_numpy()

# 进行预测
predictions = model.predict(input_data_array)

# 获取预测结果
# 假设模型输出是一个41维的one-hot向量，可以使用np.argmax()获取概率最大的类别
predicted_classes = np.argmax(predictions, axis=1) + 1

# 打印预测结果
print("Predicted classes:", predicted_classes)

# 将预测结果保存到 prediction_results 中
prediction_results = {
    'label': labels.tolist(),
    'pred': predicted_classes.tolist()
}

print(prediction_results)





# 创建 DataFrame
df = pd.DataFrame(prediction_results)

# 计算每个预测值与标签值之间的距离
distances = []
for pred, label in zip(prediction_results['pred'], prediction_results['label']):
    pred_coord = map_number_to_coordinate(pred)
    label_coord = map_number_to_coordinate(label)
    distance = calculate_distance(pred_coord, label_coord)
    distances.append(distance)

# 将距离添加到 DataFrame 中
df['distance'] = distances
mean_distance_error = np.mean(distances)
print("Mean distance error:", mean_distance_error)
# 打印 DataFrame
print(df)

# 指定要保存的 CSV 文件路径
csv_file_path = 'prediction_results_with_distance.csv'

# 将 DataFrame 写入到 CSV 文件中
df.to_csv(csv_file_path, index=False)
