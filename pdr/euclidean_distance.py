import pandas as pd
import math
import shutil

def euclidean_distance(x1, y1, x2, y2):
    """
    計算兩點間的歐式距離
    Args:
        x1, y1, z1: 第一點的 x, y, z 座標
        x2, y2, z2: 第二點的 x, y, z 座標
    Returns:
        歐式距離
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_and_add_euclidean_distance(input_file_path, output_file_path):
    """
    計算歐式距離並加入 DataFrame 中，並將更新後的資料寫入複製的 CSV 檔案
    Args:
        input_file_path: 輸入的 CSV 檔案路徑
        output_file_path: 輸出的 CSV 檔案路徑
    """
    # 讀取 CSV 檔案並轉換為 DataFrame
    df = pd.read_csv(input_file_path)
    
    # 初始化歐式距離列表，初始值為 0
    euclidean_distances = [0]

    # 計算歐式距離並加入列表
    for i in range(1, len(df)):
        distance = euclidean_distance(df.loc[i, 'pos_x'], df.loc[i, 'pos_y'],
                                       df.loc[i-1, 'pos_x'], df.loc[i-1, 'pos_y'])
        euclidean_distances.append(distance)

    # 將歐式距離列表轉為 DataFrame
    df_euclidean_distances = pd.DataFrame({'Distance': euclidean_distances})

    # 合併歐式距離 DataFrame 到原始 DataFrame
    df = pd.concat([df, df_euclidean_distances], axis=1)

    # 將更新後的資料寫入複製的 CSV 檔案
    df.to_csv(output_file_path, index=False)

    print('歐式距離已計算並寫入複製的 CSV 檔案。')

# # 呼叫函式計算歐式距離並將更新後的資料寫入複製的 CSV 檔案
# input_file_path = 'data_publish_v2\\ruixuan_body1\\processed\\data.csv'
# output_file_path_copy = 'C:\\Users\\HONG\\Desktop\\test\\58.csv'  # 複製的目標 CSV 檔案路徑

# calculate_and_add_euclidean_distance(input_file_path, output_file_path_copy)
import os

# 指定資料夾路徑
folder_path = 'C:\\Users\\HONG\\Desktop\\test\\data_publish_v2\\'

# 使用os模組的listdir函數列出資料夾中的檔案名稱
files = os.listdir(folder_path)
i=0
# 打印檔案名稱
for file in files:
    i=i+1
    input_file_path = "data_publish_v2\\{}\\processed\\data.csv".format(file)
    output_file_path_copy = "C:\\Users\\HONG\\Desktop\\test\\distance\\{}.csv".format(i)
    calculate_and_add_euclidean_distance(input_file_path, output_file_path_copy)
