import numpy as np
import pandas as pd

data = pd.read_csv("data/pems08/pems08.csv", delimiter=",", header=0)
data = data.iloc[:, 1:]
data = data.to_numpy()

no_missing = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j] == 0:
            continue
        no_missing.append(data[i,j])

no_missing = np.array(no_missing)

mean = np.mean(no_missing)
std = np.std(no_missing)
data = (data-mean) / std


points_per_hour = 12
hours = 48
points_per_sequence = hours * points_per_hour

num_complete_sequences = data.shape[0] // points_per_sequence

data = data[:num_complete_sequences * points_per_sequence]

data = data.reshape(-1, points_per_sequence, 170)
data = data.reshape(-1, hours, points_per_hour, 170)
data = data.reshape(-1, 48, 170)


total_samples = data.shape[0]
train_samples = int(total_samples * 0.7)
valid_samples = int(total_samples * 0.1)
test_samples = total_samples - train_samples - valid_samples

train_data = data[:train_samples]
valid_data = data[train_samples:train_samples + valid_samples]
test_data = data[train_samples + valid_samples:]

train_data_2d = train_data.reshape(-1, 170)
valid_data_2d = valid_data.reshape(-1, 170)
test_data_2d = test_data.reshape(-1, 170)

print("\n数据集划分:")
print(f"训练集形状: {train_data.shape}")
print(f"验证集形状: {valid_data.shape}")
print(f"测试集形状: {test_data.shape}")


