import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

import tensorflow as tf
import math

# 步骤1: 加载测试数据
test_data = pd.read_csv('120cm\\0424_wu.csv')  # 加载测试数据，格式类似于训练数据
print (test_data)
timesteps = 100  # 与训练时相同的窗口长度
features = 6  # 与训练时相同的特征数量
data_windows = []
ori=[]


for i in range(0, len(test_data) - timesteps, timesteps):
    seq = test_data[['worldXgyro', 'worldYgyro', 'worldZgyro',' worldXAccel' , 'worldYAccel', 'worldZAccel']].values[i:i+timesteps]  # 根據實際特徵名稱調整
    data_windows.append(seq)
    o = test_data[['Yaw']].values[i+timesteps] + 180  #yaw(轉向)
    ori.append(o)


model = load_model('checkpoints\\four_layers_data_argumentation_32.h5') 


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
        21: (5, 0),
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
        36: (3.0, 5),
        37: (3.6, 6),
        38: (4.2, 5),
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

# 二維陣列 x 和 y 的前三個值
x_y_values = [
    [6, 0],
    [6, 3],
    [3, 6],
    [0, 3],
    [0, 0]
]
#ori = YAW  prediction為每秒移動的距離 prev_status為前一秒使用者地朝向 zero_count為連續禁止了幾秒
def adjust_and_check_coordinates_with_values(x, y, ori, prediction, prev_status, x_y_values, count, zero_count):
    #Determine Stillness
    for pred in prediction:
        if pred == 0:
            zero_count += 1
        else:
            zero_count = 0

        if zero_count == 8:
            if count < 5:
                x, y = x_y_values[count][0], x_y_values[count][1]
                count += 1
                zero_count = 0
                status = prev_status
        #根據YAW判斷使用者往哪個方向前進
        else:
            if 55 <= ori <= 145:
                x -= pred  
                status = 1
            elif 235 <= ori <= 333:
                x += pred  
                status = 2
            elif 145 <= ori <= 235:
                y += pred  
                status = 3
            else:
                y -= pred  
                status = 4

            x = min(x, 6)
            y = min(y, 6)
            x = max(x, 0)
            y = max(y, 0)

            changed_status = status != prev_status

            if changed_status:
                min_distance = float('inf')
                nearest_position = (0, 0)
                #轉角的座標
                specified_positions = [(0, 0), (3, 0), (6, 0), (0, 6), (3, 6), (6, 6)]
                #找距離最近的轉角
                for pos in specified_positions:
                    distance = ((x - pos[0]) ** 2 + (y - pos[1]) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_position = pos

                x, y = nearest_position

    return x, y, status, count, zero_count

#將座標轉換成Reference point
def calculate_output(x, y):
    quotient_x = x // 0.6
    quotient_y = y // 0.6
    
    if quotient_x == 0 and quotient_y <= 9:
        return quotient_y + 1
    elif quotient_x == 5 and quotient_y <= 9:
        return quotient_y + 11
    elif quotient_x == 10 and quotient_y <= 9:
        return quotient_y + 21
    elif quotient_x <= 9 and quotient_y == 10:
        return quotient_x + 31
    elif quotient_x == 10 and quotient_y == 10:
        return 41
    else:
        return "Invalid input"


def IMU_offline_predict(data_windows, ori, timesteps, features, model):
    x = 6
    y = 0
    prev_status = 0
    count = 1
    zero_count = 0
    model_input = np.array(data_windows)
    model_input = model_input.reshape(-1, timesteps, features)
    predictions = model.predict(model_input)

    for i in range(len(predictions)):
        if predictions[i] < 0.1:
            predictions[i] = 0
        # 初始化 status
        status = 0
        x, y, status, count, zero_count = adjust_and_check_coordinates_with_values(x, y, ori[i], predictions[i], prev_status, x_y_values, count, zero_count)
        index = calculate_output(x, y)
        print(index, ori[i], predictions[i])
        prev_status = status  # 更新 prev_status
    results = pd.DataFrame({'Predicted Distance': predictions.flatten()})
    results.to_csv('test_results.csv', index=False)

IMU_offline_predict(data_windows, ori, timesteps, 6, model)



