import pandas as pd
import os

# 指定包含CSV檔案的資料夾路徑
folder_path = 'C:\\Users\\HONG\\Desktop\\IMU\\test\\pdr\\data\\1226\\140cm'

# 起始檔案編號
start_file_number = 122

# 初始化檔案編號計數器
file_number = start_file_number

# 使用迴圈讀取資料夾中的每個CSV檔案
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        # 讀取CSV檔案
        df = pd.read_csv(file_path)

        # 新增名為"distance"的欄位，並將所有值設為0.3
        df['Distance'] = 0.014

        # 產生新的檔案名稱
        new_file_name = f"{file_number}.csv"
        
        # 在新文件名的路径上加上 '1201_60_new/'
        new_file_path = os.path.join('data\\1226\\100hz', new_file_name)

        # 將修改後的DataFrame寫回原始檔案
        df.to_csv(new_file_path, index=False)
        
        # 增加檔案編號計數器
        file_number += 1
# 此檔案為用來label不同長度