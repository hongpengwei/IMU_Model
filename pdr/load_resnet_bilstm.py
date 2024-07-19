import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.models import load_model
import matplotlib.pyplot as plt  # 引入Matplotlib
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


import tensorflow as tf


    
# 定义 Adam 优化器并设置学习率
optimizers = tf.keras.optimizers.Adam(learning_rate=0.001)

# 定义 ReduceLROnPlateau 学习率调度器
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, verbose=True, epsilon=1e-12)

# 修改這一行，指定新的資料夾路徑
folder_path = '1226\\train'

# 使用迴圈讀取資料夾中的每個CSV檔案
sequence_data = []
distances = []
sequence_length = 200
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 組合完整的檔案路徑
        file_path = os.path.join(folder_path, file_name)

        # 載入CSV檔案至Pandas DataFrame
        data = pd.read_csv(file_path)

        # 初始化空的序列資料和距離列表
        for i in range(0, len(data) - sequence_length, sequence_length):
            distance = np.sum(data['Distance'].values[i:i+sequence_length])
            distances.append(distance)

        # 分割資料成 200 筆一組的序列
        for i in range(0, len(data) - sequence_length, sequence_length):
            # seq = data[['worldZAccel']].values[i:i+sequence_length]
            seq = data[['worldXgyro', 'worldYgyro', 'worldZgyro', ' worldXAccel', 'worldYAccel', 'worldZAccel']].values[i:i+sequence_length]
            sequence_data.append(seq)
# 轉換成 NumPy 陣列
X = np.array(sequence_data)
y = np.array(distances)

# 分割成訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Specify the paths to save the best model checkpoints for each batch size
checkpoint_path_32 = 'checkpoints\\resnet_bilstm_two_layers_32_fine_tune.h5'
# checkpoint_path_64 = 'checkpoints/best_1207_64_new.h5'

# Define the ModelCheckpoint callbacks for each batch size
model_checkpoint_32 = ModelCheckpoint(
    checkpoint_path_32,
    save_best_only=True,
    verbose=1
)

# model_checkpoint_64 = ModelCheckpoint(
#     checkpoint_path_64,
#     save_best_only=True,
#     verbose=1
# )

# Load the pre-trained models for each batch size
model_32 = load_model('checkpoints\\resnet_bilstm_two_layers_32.h5')
# model_64 = load_model('checkpoints/best_model_1204_64.h5')
model_32.summary()
# Compile the models
# optimizers = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
model_32.compile(loss='mean_squared_error', optimizer=optimizers)
# model_64.compile(loss='mean_absolute_error', optimizer='adam')

# Train the models with different batch sizes
history_32 = model_32.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[model_checkpoint_32, scheduler])
# history_64 = model_64.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[model_checkpoint_64])

# Save the updated models
model_32.save('resnet_bilstm_two_layers_fine_tune_32.h5')
# model_64.save('best_model_test_64.h5')

# Merge training histories for plotting
history_merged = {
    '32_train_loss': history_32.history['loss'],
    '32_val_loss': history_32.history['val_loss'],
    # '64_train_loss': history_64.history['loss'],
    # '64_val_loss': history_64.history['val_loss']
}

# Plot the training history
plt.plot(history_merged['32_train_loss'], label='32_train')
plt.plot(history_merged['32_val_loss'], label='32_validation')
# plt.plot(history_merged['64_train_loss'], label='64_train')
# plt.plot(history_merged['64_val_loss'], label='64_validation')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training and Validation Loss for Batch Sizes 32(Resneet_BILSTM)')
plt.grid(True)

# Save the plot
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/resnet_bilstm_two_layers_32_fine_tune.png')

# Display the plot if needed
plt.show()