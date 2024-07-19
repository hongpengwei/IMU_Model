import pandas as pd
import os

# 指定包含CSV檔案的資料夾路徑
folder_path = 'C:\\Users\\HONG\\Desktop\\IMU\\test\\pdr\\data\\before'

# 使用迴圈讀取資料夾中的每個CSV檔案
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 組合完整的檔案路徑
        file_path = os.path.join(folder_path, file_name)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 去除重复的时间戳
        data = data.drop_duplicates(subset=['Timestamp'])

        # 将第一列解析为日期时间格式
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y%m%d %H:%M:%S')

        # 将时间戳列设置为索引
        data.set_index('Timestamp', inplace=True)

        # 重新采样为200Hz，执行线性插值
        resampled_data = data.resample('5ms').interpolate(method='linear')

        # 打印插值后的数据
        print(resampled_data)

        resampled_data.index = resampled_data.index.strftime('%H:%M:%S')

        # 构建新的文件名，将原文件名中的 '.csv' 替换为 '_new.csv'
        new_file_name = file_name.replace('.csv', '_new.csv')

        # 在新文件名的路径上加上 '1129_6/'
        new_file_path = os.path.join('data\\after', new_file_name)

        # 将插值后的数据保存到新的CSV文件中
        resampled_data.to_csv(new_file_path)
