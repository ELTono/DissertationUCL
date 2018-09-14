# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:32:23 2018

@author: Antonio
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import keras
from keras.callbacks import ReduceLROnPlateau,TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import pickle

#############################################################
'             Fitting and training the model                '
#############################################################
####### Model parameters ########
# time seed
model = 'FCN'
sub_model = 'leaky_ensemble'
np.random.seed(813306) 
filter_list = [128,256,128] ## For wavenet you need to use 4 filters for other only 3 filters   
k_size_list = [8,5,3]
k_size = 2 #  This parameter is for the wavenet architecture
conv_activation = 'relu'
dense_activation = 'softmax'
n_epochs = 10
batch_size = 3000
window_size = 100
lr = 1e-5
alpha = 0.3 # Alpha parameter for FCN Leaky ReLU
reg_param = 0.001 # Regularization parameter for FCN model
n_features = final_x_train.shape[-1]
output_size = y_train.shape[1]
model_name = model_selection + '_'+ sub_model + '_standard_epoch{}_batch{}'.format(n_epochs,batch_size) 
log_name = 'FCN_leaky_ensemble'

optimizer = Adam(lr)

model = call_model(model,sub_model,filter_list, k_size_list,k_size,conv_activation,
                   dense_activation, window_size, n_features, output_size,optimizer,alpha,reg_param,
                   summary = True)
                
callbacks_list = make_callbacks(model_selection,sub_model,log_name
                           ,n_epochs,reduce_lr = True,tensor_board = False,checkpoint = False)

train_gen = TimeseriesGenerator(final_x_train,y_train,length = window_size,sampling_rate = 1,
                               batch_size = batch_size)

val_gen = TimeseriesGenerator(final_x_val,y_val,length = window_size,sampling_rate = 1,
                               batch_size = batch_size)

hist = model.fit_generator(train_gen,epochs = n_epochs,verbose = 1,validation_data = val_gen,
                           callbacks = callbacks_list,shuffle = True,
                           use_multiprocessing= True)
########################################################
'       Storing and plotting the results               '
########################################################
acc = hist.history['categorical_accuracy']
val_acc = hist.history['val_categorical_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
  
ave_acc, ave_val_acc = np.average(acc), np.average(val_acc)
std_acc, std_val_acc = np.std(acc), np.std(val_acc)
ave_loss, ave_val_loss = np.average(loss), np.average(val_loss)
std_loss, std_val_loss = np.std(loss), np.std(val_loss)

make_plots(model_name,acc, val_acc, loss, val_loss,n_epochs,batch_size)

##########################################################
'           Save the model and results                   '
##########################################################
path = './'+model_selection+'/'+sub_model+'/' ## You need to check manually this setting
pkl_name =   path + model_selection+'_epoch{}_batch{}'.format(n_epochs,batch_size) + '.pickle'
with open(pkl_name, 'wb') as f:
    pickle.dump([acc, val_acc, loss, val_loss, ave_acc, ave_val_acc, std_acc,
                 std_val_acc, ave_loss, ave_val_loss,  std_loss, std_val_loss],
                f, protocol=-1)
    print('Per-iteration results saved as: ' ,pkl_name)
    
save_model(model_selection,sub_model,model,model_name)

##########################################################
'           Load the model and results                   '
##########################################################
# Storing the variables in a pickle file
# loading the variables

file_name = 'FCN_dilation_epoch10_batch3000.pickle'
pkl_name = './'+ model_selection+'/' + sub_model + '/' + file_name
model_name = 'FCN_dilation_epoch10_batch3000'
with open(pkl_name, 'rb') as f:
      acc, val_acc, loss, val_loss, ave_acc, ave_val_acc, std_acc, std_val_acc, ave_loss, ave_val_loss,  std_loss, std_val_loss = pickle.load(f)

make_plots(model_name,acc,val_acc,loss,val_loss,n_epochs,batch_size)
