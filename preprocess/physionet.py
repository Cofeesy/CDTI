import numpy as np
from pypots.data import load_specific_dataset


data = load_specific_dataset('physionet_2012')
print(data.keys())

train_X = data['train_X']
val_X = data['val_X']
test_X = data['test_X']

train_X = train_X.reshape(-1,37)
train_X[np.isnan(train_X)] = -200
val_X = val_X.reshape(-1,37)
val_X[np.isnan(val_X)] = -200
test_X = test_X.reshape(-1,37)
test_X[np.isnan(test_X)] = -200


