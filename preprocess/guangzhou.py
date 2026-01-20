import numpy as np
import pandas as pd

# Download traffic_speed_data.csv in to "./Data" from https://zenodo.org/record/1205229


data = pd.read_csv("data/guangzhou/Guangzhou.csv", delimiter=",", header=0)
data = data.replace(np.nan, -200).to_numpy()

miss_p = np.where(data == -200)
no_missing = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j] == -200:
            continue
        no_missing.append(data[i,j])

no_missing = np.array(no_missing)

mean = np.mean(no_missing)
std = np.std(no_missing)
data = (data-mean) / std
data[miss_p] = -200


data = data.reshape(-1, 48, 214)


total_days = data.shape[0]  
train_days = int(total_days * 0.7)  
valid_days = int(total_days * 0.1)  
test_days = total_days - train_days - valid_days 


train_data = data[:train_days]  
valid_data = data[train_days:train_days + valid_days]  
test_data = data[train_days + valid_days:]  

train_data_2d = train_data.reshape(-1, 214)
valid_data_2d = valid_data.reshape(-1, 214)
test_data_2d = test_data.reshape(-1, 214)

np.savetxt("data/guangzhou/guangzhou_norm_train.csv", train_data_2d, delimiter=",", fmt="%6f")
np.savetxt("data/guangzhou/guangzhou_norm_valid.csv", valid_data_2d, delimiter=",", fmt="%6f")
np.savetxt("data/guangzhou/guangzhou_norm_test.csv", test_data_2d, delimiter=",", fmt="%6f")






