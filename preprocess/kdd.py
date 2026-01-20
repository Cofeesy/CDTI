import pandas as pd
import numpy as np


data = pd.read_csv("data/kdd/kdd.csv", delimiter=",", header=0)
data = data.replace(np.nan, -200).to_numpy()


list = []
for i in range(9):
    station_record = data[:, i*13+2: (i+1)*13] 
    list.append(station_record)
data = np.stack(list, axis=1).reshape(data.shape[0], -1)

# 标准化处理
means, stds = [], []
for j in range(data.shape[1]):
    data_j = []
    for i in range(data.shape[0]):
        if data[i,j] == -200:
            continue
        data_j.append(data[i,j])
    data_j = np.array(data_j)
    mean_j = np.mean(data_j)
    std_j = np.std(data_j)

    for i in range(data.shape[0]):
        if data[i,j] == -200:
            continue
        data[i,j] = (data[i,j] - mean_j) / std_j
    means.append(mean_j)
    stds.append(std_j)

data = data[0:8016]
print(data.shape) 

data = data.reshape(-1, 48, 99)
print(data.shape)  

total_days = data.shape[0]  
train_days = int(total_days * 0.7)  
valid_days = int(total_days * 0.1)  #
test_days = total_days - train_days - valid_days  

# 根据时间维度划分数据
train_data = data[:train_days]  
valid_data = data[train_days:train_days + valid_days]  
test_data = data[train_days + valid_days:]  


train_data_2d = train_data.reshape(-1, 99)
valid_data_2d = valid_data.reshape(-1, 99)
test_data_2d = test_data.reshape(-1, 99)

np.savetxt("data/kdd/kdd_norm_train.csv", train_data_2d, delimiter=",", fmt="%6f")
np.savetxt("data/kdd/kdd_norm_valid.csv", valid_data_2d, delimiter=",", fmt="%6f")
np.savetxt("data/kdd/kdd_norm_test.csv", test_data_2d, delimiter=",", fmt="%6f")