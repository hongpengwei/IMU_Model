import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

import tensorflow as tf


# 步骤1: 加载测试数据
test_data = pd.read_csv('data\\after\\0625_105_new.csv')  # 加载测试数据，格式类似于训练数据
print (test_data)


# 步骤3: 创建数据窗口
# 你的数据窗口创建可能需要根据模型输入的形状进行调整
timesteps = 200  # 与训练时相同的窗口长度
features = 6  # 与训练时相同的特征数量
data_windows = []
ori=[]

for i in range(0, len(test_data) - timesteps, timesteps):
    seq = test_data[['worldXgyro', 'worldYgyro', 'worldZgyro',' worldXAccel' , 'worldYAccel', 'worldZAccel']].values[i:i+timesteps]
    # seq = test_data[['worldXgyro', 'worldYgyro', 'worldZgyro',' worldXAccel' , 'worldYAccel', 'worldZAccel']].values[i:i+timesteps]  # 根據實際特徵名稱調整
    data_windows.append(seq)
    o = test_data[['Yaw']].values[i+timesteps]   #yaw(轉向)
    ori.append(o)

# 步骤4: 准备模型输入
model_input = np.array(data_windows)
model_input = model_input.reshape(-1, timesteps, features)

# 步骤5: 加载已训练的模型
model = load_model('checkpoints\\four_layers_bilstm_model2_32.h5')  # 替换成你训练好的模型的文件名



# 步骤6: 使用模型进行预测
predictions = model.predict(model_input)
i=0
for i in range(len(predictions)):
    if(predictions[i]<0.05):
        predictions[i]=0
# 步骤7: 可以根据需要对预测结果进行进一步处理和可视化

# 例如，你可以将预测结果保存到一个CSV文件：
results = pd.DataFrame({'Predicted Distance': predictions.flatten()})
results.to_csv('test_results_4.csv', index=False)


# 也可以绘制预测结果的图表：
plt.figure(figsize=(14, 6))

# Plot Predicted Distance
plt.subplot(1, 2, 1)
plt.plot(test_data.index[timesteps - 1:timesteps - 1 + len(predictions)]-98, predictions, label='Predicted Distance')
plt.xlabel('Time')
plt.ylabel('Predicted Distance')
plt.legend()
plt.title('Predicted Distance vs Time')
plt.grid(True)

# Plot Yaw Angle
plt.subplot(1, 2, 2)
plt.plot(test_data.index[timesteps - 1:timesteps - 1 + len(predictions)]-98, ori, label='Yaw Angle', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Yaw Angle')
plt.legend()
plt.title('Yaw Angle vs Time')
plt.grid(True)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
plt.savefig('predict_horizontal.png')