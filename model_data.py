# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 05:24:34 2018

@author: Antonio
"""

'''
This code makes the data preparation for the main analysis done in model_run.py. This code
read the data from the file path, makes the fix to the required variables, makes the train-test split and
scales the features.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

###############################################################################
'                        Preparing the data                                   '
###############################################################################
file_path = 'C:\\Users\\Antonio\\Desktop\\Msc CSML\\Thesis Project\\Code'
os.chdir(file_path)
from model_functions import *

### Reading the data
file_name = 'total_data.csv'
financial_data = pd.read_csv('./Data/'+file_name)
financial_data = financial_data.iloc[:,2:]

## We are going to apply logarithm to some features that look skewed to the left.
financial_data['ADX200R'] = np.log1p(financial_data['ADX200R'])
financial_data['ADX500R'] = np.log1p(financial_data['ADX500R'])

acc_swing = financial_data['ACC_SWING_INDEX_MPC_1'].values
neg_sum = np.sum(acc_swing < 0)
acc_swing[acc_swing < 0] = 0.001

financial_data['ACC_SWING_INDEX_MPC_1'] = np.sqrt(acc_swing)
###############################################################################
'               Spliting data in train and validation sets                    '
###############################################################################
train_portion = 0.7
val_portion = 0.2
test_portion = 0.1
n_samples = financial_data.shape[0]

train_size = np.int(np.floor(train_portion*n_samples))
non_train_size = n_samples - train_size
val_size = np.int(np.floor(val_portion*n_samples))

x_data = financial_data.iloc[:,1:].values
y_data = financial_data.iloc[:,:1]

# We transform the target into integers and One Hot Encoding
print(y_data['label_rol'].unique()) # We check the number of classes in the target. The classes are: -1,0 and 1
y_data = y_data.values + 1
y_data = y_data.astype(int) # We transform the matrix into integer values
print(np.unique(y_data)) # We check that we fix the negative values in the classes

encoder = OneHotEncoder(sparse = False)
y_data = encoder.fit_transform(y_data)

x_train = x_data[:train_size,:]
x_val = x_data[train_size:(train_size + val_size),:]
x_test = x_data[(train_size + val_size):,:]

y_train = y_data[:train_size,:]
y_val = y_data[train_size:(train_size + val_size),:]
y_test = y_data[(train_size + val_size):,:]

###############################################################################
'                       Scaling variables                                     '
###############################################################################
scaler = general_scaler(method = 'standard')
features_scaled = scaler.fit_transform(x_train) # it is going to transform the data frame to numpy array.

# We clipp the outliers so they have the values of 2 and -2

features_clipped = np.copy(features_scaled)
features_clipped[features_scaled > 2] = 2
features_clipped[features_scaled < -2] = -2

#plot_hist(features_clipped,important_features[2:],title = 'training_data_clipped',
#          nrows = 4,ncols = 5)

### Nows lets apply the changes to the validation data
x_val_scaled = scaler.transform(x_val)

x_val_scaled[x_val_scaled > 2] = 2
x_val_scaled[x_val_scaled < -2] = -2

### Nows lets apply the changes to the test data
x_test_scaled = scaler.transform(x_test)

x_test_scaled[x_test_scaled > 2] = 2
x_test_scaled[x_test_scaled < -2] = -2

#plot_hist(x_val_scaled,important_features[2:],title = 'validation_data_clipped',
#          nrows = 4,ncols = 5)
### Finally we reshape the data so it has the form [batch_size,features,1]

final_x_train = features_clipped
final_x_val = x_val_scaled
final_x_test = x_test_scaled





















