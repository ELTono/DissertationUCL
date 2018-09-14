# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 06:57:05 2018

@author: Antonio / Model Functions
"""

'''
Code that defines all the functions used in model_data.py and model_run.py
'''

import numpy as np
from keras.layers import Input,Conv1D, Dense, GlobalAveragePooling1D, Activation,Dropout,LeakyReLU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.merge import add
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder

##################################################
'            Helper functions for data           '
##################################################

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def make_UCR_data():

    flist  = ['Beef']
    for each in flist:
        fname = each
        x_train, y_train = readucr(fname+'_TRAIN')
        x_test, y_test = readucr(fname+'_TEST')
        nb_classes = len(np.unique(y_test))
         
        y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
         
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
         
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean)/(x_train_std)
          
        x_test = (x_test - x_train_mean)/(x_train_std)
        # original code produces 4D array, change to 3D for Conv1D implemntation
        x_train = x_train.reshape(x_train.shape + (1,)) 
        x_test = x_test.reshape(x_test.shape + (1,))
        
        return x_train, Y_train, x_test, Y_test
    

def get_data(file,memory_size):
    '''
    This function reads only the file until the amount of memory defined in the parameters.
    '''
    df_ls = []
    memory_usage = 0
    for partial_data in pd.read_csv(file,
                               chunksize=5000):
        memory_usage += partial_data.memory_usage().sum()
        print(memory_usage/1000000)
        df_ls.append(partial_data)
        if memory_usage > memory_size: break
    total_data = pd.concat(df_ls)
    
    return total_data


def plot_hist(data,col_names,title,nrows,ncols):
    '''
    Helper function to create a series of histogram
    '''
    k = 0
    nrows = nrows
    ncols = ncols
    fig, axes = plt.subplots(nrows, ncols,figsize = (ncols*6,nrows*4))
    for i in range(5):
        for j in range(5):
            axes[i,j].hist(data[:,k],bins=15,color='blue')   
            axes[i,j].set_title(col_names[k]+" histogram")
            k = k+1
    fig.suptitle(title)
    plt.show()
    
    
def general_scaler(method = 'standard'):
    '''
    General function to create 3 scalers: standard, quantile and robust. All of them defined as sklearn classes
    '''
    
    if method not in ['standard','quantile','robust']:
        print('Error: the method argument must be a string pointing to one of the 3 valid methods')
        return
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution = 'normal')
    elif method == 'robust':
        scaler = RobustScaler()
        
    return scaler

###############################################################################
'                      Helper functions for models                            '
###############################################################################
    
######################### FCN models ##########################################
def create_FCN(filter_list, k_size_list,conv_activation,
                 dense_activation, window_size, n_features, output_size,optimizer,reg_param = 0 ,
                 summary = True):
    '''
    Standard FCN model from the paper "Strong baseline" adding regularization.
    '''
    reg_val = regularizers.l2(reg_param)
    x = Input(shape=(window_size,n_features))
    x_norm = BatchNormalization(axis = 1)(x)
    ## First block
    block_1 = Conv1D(padding='same',kernel_size = k_size_list[0], filters= filter_list[0],
                            kernel_regularizer= reg_val)(x_norm)
    block_1 = BatchNormalization(axis = 1)(block_1)
    block_1 = Activation(conv_activation)(block_1)
    
    ## Second block
    block_2 = Conv1D(padding='same',kernel_size = k_size_list[1], 
                     filters= filter_list[1],kernel_regularizer= reg_val)(block_1)
    block_2 = BatchNormalization(axis = 1)(block_2)
    block_2 = Activation(conv_activation)(block_2)
    
    ## Third block
    block_3 = Conv1D(padding='same',kernel_size = k_size_list[2], 
                     filters= filter_list[2],kernel_regularizer= reg_val)(block_2)
    block_3 = BatchNormalization(axis = 1)(block_3)
    block_3 = Activation(conv_activation)(block_3)
    
    ## Final layers
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation,
                   kernel_regularizer= reg_val)(full)
    #output = Dropout(0.5)(output)
    
   # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
        
    print('Model selected: FCN_baseline_regularizer:{}'.format(reg_param))
    return model

def create_FCN_leaky(filter_list, k_size_list,conv_activation,
                 dense_activation, window_size, n_features, output_size,optimizer, summary = True):
    '''
    Standard FCN model with regularization parameter and using leaky_relu instead of ReLU
    '''
    x = Input(shape=(window_size,n_features))
    ## First block
    block_1 = BatchNormalization(axis = 1)(x)
    block_1 = Conv1D(padding='same',kernel_size = k_size_list[0], filters= filter_list[0],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_1 = LeakyReLU(alpha)
    block_1 = BatchNormalization(axis = 1)(block_1)
    
    ## Second block
    block_2 = Conv1D(padding='same',kernel_size = k_size_list[1], filters= filter_list[1],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_2 = LeakyReLU(alpha)
    block_2 = BatchNormalization(axis = 1)(block_2)
    
    ## Third block
    block_3 = Conv1D(padding='same',kernel_size = k_size_list[2], filters= filter_list[2],
                            kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_3 = LeakyReLU(alpha)
    block_3 = BatchNormalization(axis = 1)(block_3)
    
    ## Final layers
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation,
                   kernel_regularizer=regularizers.l2(0.001))(full)
    #output = Dropout(0.5)(output)
    
   # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
        
    print('Model selected: FCN_leaky')
    return model


def create_FCN_dilation(filter_list, k_size_list,conv_activation,
                 dense_activation, window_size, n_features, output_size,optimizer, summary = True):
    '''
    Modification of the base model by adding a dilation rate with a factor of 2 per layer. This idea is taken from the
    paper conditional time series forecasting.
    '''
     
    x = Input(shape=(window_size,n_features))
    ## First block
    block_1 = BatchNormalization(axis = 1)(x)
    block_1 = Conv1D(padding='same',activation =  conv_activation,kernel_size = k_size_list[0],
                     filters= filter_list[0],dilation_rate = 1)(block_1)
    block_1 = BatchNormalization(axis = 1)(block_1)
    
    ## Second block
    block_2 = Conv1D(padding='same',activation =  conv_activation,
                            kernel_size = k_size_list[1], filters= filter_list[1],dilation_rate = 2)(block_1)
    block_2 = BatchNormalization(axis = 1)(block_2)
    
    ## Third block
    block_3 = Conv1D(padding='same',activation =  conv_activation,
                            kernel_size = k_size_list[2], filters= filter_list[2],dilation_rate = 4)(block_2)
    block_3 = BatchNormalization(axis = 1)(block_3)
    
    ## Final layers
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation)(full)
    
   # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
    
    print('Model selected:FCN_dilation')
    return model

def create_FCN_2_1(filter_list, k_size_list,dense_activation, window_size, 
                   n_features, output_size,optimizer, summary = True):
    '''
    FCN model with regularization and 2:1 ConvNet ReLU ratio based in the paper: "Training better CNNs requires to rethink
    ReLU"
    '''
    x = Input(shape=(window_size,n_features))
    ############# First block ################
    block_1 = BatchNormalization(axis = 1)(x)
    ### First pair
    block_1 = Conv1D(padding='same',kernel_size = k_size_list[0], filters= filter_list[0],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_1 = BatchNormalization(axis = 1)(block_1)
    ### Second pair
    block_1 = Conv1D(padding='same',kernel_size = k_size_list[0], filters= filter_list[0],
                            kernel_regularizer=regularizers.l2(0.001))(x)
    block_1 = BatchNormalization(axis = 1)(block_1)
    ### Activation function
    block_1 = Activation('relu')(block_1)
    
    ############## Second block ################
    ### First pair
    block_2 = Conv1D(padding='same',kernel_size = k_size_list[1], filters= filter_list[1],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_2 = BatchNormalization(axis = 1)(block_2)
    ### Second pair
    block_2 = Conv1D(padding='same',kernel_size = k_size_list[1], filters= filter_list[1],
                            kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_2 = BatchNormalization(axis = 1)(block_2)
    ### Activation function
    block_2 = Activation('relu')(block_2)
    
    ############## Third block ################
    ### First pair
    block_3 = Conv1D(padding='same',kernel_size = k_size_list[2], filters= filter_list[2],
                            kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_3 = BatchNormalization(axis = 1)(block_3)
    ### Second pair
    block_3 = Conv1D(padding='same',kernel_size = k_size_list[2], filters= filter_list[2],
                            kernel_regularizer=regularizers.l2(0.001))(block_3)
    block_3 = BatchNormalization(axis = 1)(block_3)
    ### Activation function
    block_3 = Activation('relu')(block_3)

    ## Final layers
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation,
                   kernel_regularizer=regularizers.l2(0.001))(full)
    #output = Dropout(0.5)(output)
    
   # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
        
    print('Model selected: FCN_2_1')
    return model

def create_FCN_leaky_ensemble(filter_list, k_size_list,conv_activation,
                 dense_activation, window_size, n_features, output_size,optimizer, summary = True):
    '''
    FCN model with regularization and a Leaky ReLU ensemble architecture based in the paper: "Training Better CNNs requires
    rethink ReLU
    '''
    x = Input(shape=(window_size,n_features))
    ############# First block ################
    block_1 = BatchNormalization(axis = 1)(x)
    # First pair
    block_1 = Conv1D(padding='same',kernel_size = k_size_list[0], filters= filter_list[0],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_1 = BatchNormalization(axis = 1)(block_1)
    activation_1 = Activation(conv_activation)(block_1)
    
    # Residual conecction
    block_1 = add([block_1,activation_1])
    ############## Second block ################
    ### First pair
    block_2 = Conv1D(padding='same',kernel_size = k_size_list[1], filters= filter_list[1],
                            kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_2 = BatchNormalization(axis = 1)(block_2)
    activation_2 = Activation(conv_activation)(block_2)
    # Residual conecction
    block_2 = add([block_2,activation_2])
    
    ############## Third block ################
    ### First pair
    block_3 = Conv1D(padding='same',kernel_size = k_size_list[2], filters= filter_list[2],
                            kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_3 = BatchNormalization(axis = 1)(block_3)
    activation_3 = Activation(conv_activation)(block_3)
    # Residual conecction
    block_3 = add([block_3,activation_3])
    
    ############# Final layers ################
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation,
                   kernel_regularizer=regularizers.l2(0.001))(full)
    #output = Dropout(0.5)(output)
    
   # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
        
    print('Model selected: FCN_leaky_ensemble')
    return model

########################### Wavenet Model #####################################
def wavenet_block(x_input,input_dim,n_filter=1,k_size = 2,activation='relu',
                  ini_net = False,d_rate=1): 
   '''
   This function implement the basic block for the wavenet architecture. 
   '''
   # If the the block is the beginning of the network then we normalize the input if not we use the input as it is.
   if ini_net:
        bn_ini = BatchNormalization()(x_input)
   else:
        bn_ini = x_input
   # Block
   conv = Conv1D(n_filter, k_size, padding='causal',dilation_rate = d_rate)(bn_ini)
   bn = BatchNormalization(axis=1)(conv)
   output = Activation(activation)(bn)
   
   # implement skip connection:
   bottleneck = not (input_dim[-1] == n_filter)  
   if bottleneck:
       skip = Conv1D(n_filter, kernel_size = 1, padding='same')(x_input)
       skip = BatchNormalization()(skip)
   else:
       skip = BatchNormalization()(x_input)
        
   final_output = add([skip, output]) #padding needs to be same for merge to work 
   final_output = Activation(activation)(final_output)
   
   return final_output

def create_wavenet(filter_list,k_size,conv_activation,dense_activation, window_size,
                   n_features, output_size,optimizer, summary = True):
    '''
    Wavenet structure based in the paper: "Conditional Time Series Forecasting with Convolutional Neural Networks"
    '''
    
    x_dim = (window_size,n_features)
    x = Input(shape=x_dim)
    ## First wavenet block
    block_1 = wavenet_block(x,x_dim,filter_list[0],k_size,activation = conv_activation,
                            ini_net = True,d_rate=1)
    ## Second wavenet block
    block_1_dim = (window_size,filter_list[0])
    block_2 = wavenet_block(block_1,block_1_dim,filter_list[1],k_size,activation = conv_activation,
                            ini_net = False,d_rate=2)
    ## Third wavenet block
    block_2_dim = (window_size,filter_list[1])
    block_3 = wavenet_block(block_2,block_2_dim,filter_list[2],k_size,activation = conv_activation,
                            ini_net = False,d_rate=4)
    ## Fourth wavenet block
    block_3_dim = (window_size,filter_list[2])
    block_4 = wavenet_block(block_3,block_3_dim,filter_list[3],k_size,activation = conv_activation,
                            ini_net = False,d_rate=8)
     ## Final layers
    full = Conv1D(filters = 1, kernel_size = 1, padding='same')(block_4)
    output = Dense(output_size, activation= dense_activation)(full)
    
    # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
    print('Model selected:Wavenet baseline')
    return model
########################### ResNet model#######################################
def resnet_block(x_input,input_dim,n_filter=1,k_size_list=[8,5,3],padding="same",activation='relu',ini_net = False):
   '''
   This function implement the basic block for the resnet architecture. It is composed first for a batch normalization
   of the input, then a block of one convolutional followed by a batch normalization. This is repeated 3 times with
   different kernel size in each repetition. It is ended with a skip connection to sum the input with the output of the
   convolution. It is based in the paper: "Time series classification from scratch with deep neural networks: A strong
   baseline"
   '''
   # If the the block is the beginning of the network then we normalize the input if not we use the input as it is.
   if ini_net:
       bn_ini = BatchNormalization()(x_input)
   else:
       bn_ini = x_input
   
   # First block
   conv_1 = Conv1D(n_filter, k_size_list[0], padding=padding)(bn_ini)
   bn_1 = BatchNormalization(axis=1)(conv_1)
   output_1 = Activation(activation)(bn_1)
   
   # Second block
   conv_2 = Conv1D(n_filter, k_size_list[1], padding= padding)(output_1)
   bn_2 = BatchNormalization(axis=1)(conv_2)
   output_2 = Activation(activation)(bn_2)
   
   #Third block
   conv_3 = Conv1D(n_filter, k_size_list[2], padding= padding)(output_2)
   bn_3 = BatchNormalization(axis=1)(conv_3)
   
# implement skip connection:
   bottleneck = not (input_dim[-1] == n_filter)  
   if bottleneck:
       skip = Conv1D(n_filter, 1, padding='same')(x_input)

       skip = BatchNormalization()(skip)
   else:
       skip = BatchNormalization()(x_input)
        
   final_output = add([skip, bn_3]) #padding needs to be same for merge to work 
   final_output = Activation('relu')(final_output)
   
   return final_output
    
def create_resnet(filter_list, k_size_list,conv_activation,
                 dense_activation, window_size, n_features, output_size,optimizer, summary = True):
    '''
    This functino implements the resnet architecture presented in:"Time series classification from scratch with deep neural networks: A strong
   baseline" 
    '''
    
    x_dim = (window_size,n_features)
    x = Input(shape=x_dim)
    ## First resnet block
    block_1 = resnet_block(x,x_dim,filter_list[0],k_size_list,activation = conv_activation,ini_net = True)
    
    ## Second block
    block_1_dim = (window_size,filter_list[0])
    block_2 = resnet_block(block_1,block_1_dim,filter_list[1],k_size_list,activation = conv_activation)
    
    ## Third block
    block_2_dim = (window_size,filter_list[1])
    block_3 = resnet_block(block_2,block_2_dim,filter_list[2],k_size_list,activation = conv_activation)
    
    ## Final layers
    full = GlobalAveragePooling1D()(block_3)  
    output = Dense(output_size, activation= dense_activation)(full)
    
    # We set the model
    model = Model(inputs = x,outputs = output)
    # We define the optimizer and compile the model
    optimizer = optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    if summary:
        model.summary()
    print('Model selected:Resnet baseline')
    return model

def make_callbacks(folder,sub_folder,name,n_epochs, reduce_lr = False, tensor_board = True,checkpoint = True):
    '''
    Function that create the callbacks for the checkpoint, tensorboard files and reduce of the learning rate.
    '''
    
    callbacks_ls = []
    
    if reduce_lr == True:
       red = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience=300, min_lr=5e-7) 
       callbacks_ls.append(red)
    if checkpoint == True:
        
       weights_file= './' + folder + '/'+ sub_folder + '/logs/'+"best_weights"+ "_epoch_{epoch:02d}_val_acc_{val_categorical_accuracy:.2f}.hdf5"
       print(weights_file)
       check = ModelCheckpoint(weights_file, monitor='val_categorical_accuracy',
                               save_best_only= False, mode= 'max', period = 1)                           
       callbacks_ls.append(check) 
    if tensor_board == True:
        
       path = './'+ folder+'/'+ sub_folder + '/logs/' + name
       tens = TensorBoard(log_dir = path, histogram_freq= n_epochs)
       callbacks_ls.append(tens)
    
    return callbacks_ls


##################### Master model function #############################################
def call_model(model,sub_model,filter_list,k_size_list,k_size,conv_activation,dense_activation,
               window_size,n_features,output_size,optimizer,alpha,reg_param,summary):
    '''
    This is a helper function to call the final model.
    '''
    if model == 'FCN' and sub_model == 'baseline' :
    
        model = create_FCN(filter_list, k_size_list,conv_activation,
                    dense_activation, window_size, n_features, output_size,optimizer,reg_param, 
                    summary = summary)
    
    elif model == 'FCN' and sub_model == 'dilation':
    
        model = create_FCN_dilation(filter_list, k_size_list,conv_activation,
                    dense_activation, window_size, n_features, output_size,optimizer, summary = summary)
        
    elif model == 'FCN' and sub_model == 'leaky_relu':
    
        model = create_FCN_leaky(filter_list, k_size_list,conv_activation,
                    dense_activation, window_size, n_features, output_size,optimizer,alpha, summary = summary)
    
    elif model == 'FCN' and sub_model == '2_1':
    
        model = create_FCN_2_1(filter_list, k_size_list,dense_activation, 
                                 window_size, n_features, output_size,optimizer,summary = summary)
        
    elif model == 'FCN' and sub_model == 'leaky_ensemble':
    
        model = create_FCN_leaky_ensemble(filter_list, k_size_list,conv_activation,dense_activation, 
                                 window_size, n_features, output_size,optimizer,summary = summary)

    elif model == 'Resnet' and sub_model == 'baseline':
    
        model = create_resnet(filter_list, k_size_list,conv_activation,
                    dense_activation, window_size, n_features, output_size,optimizer, summary = summary)
    
    elif model == 'wavenet' and sub_model == 'baseline':
        
        model = create_wavenet(filter_list, k_size,conv_activation,
                    dense_activation, window_size, n_features, output_size,optimizer, summary = summary)
        
    return model

def save_model(folder,sub_folder,model, MODEL_NAME):
    
    '''
    This function save the model given a folder and subfolder. It stores the model and the weights
    using a h5 format.
    '''
    """ Function assigns dirctory and file names and saves model with json and weights using h5"""
    folder_path = './'+folder+'/'+sub_folder+'/results/'
    FILENAME_JSON = MODEL_NAME + '.json'
    FILENAME_Weights =  MODEL_NAME +'_Model_Weights'+ '.h5' 

    model_json = model.to_json()
    with open(folder_path + FILENAME_JSON, "w") as json_file:
        json_file.write(model_json)
    print("Saved JSON model to disk")

    model.save_weights(folder_path + FILENAME_Weights)
    print("Saved model weights to disk")

def make_plots(model_name,acc, val_acc, loss, val_loss,n_epochs,batch_size):
  #  "Accuracy"
    plt.plot(acc)
    plt.plot(val_acc)
    plt.axhline(y=.5, linewidth=1, color = 'k')
    plt.title(model_name + ' Accuracy n_epochs:{} batch_size:{}'.format(n_epochs,batch_size))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
  # "Loss"
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(model_name + ' Loss n_epochs:{} batch_size:{}'.format(n_epochs,batch_size))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'],  loc='upper right')
    plt.show()


