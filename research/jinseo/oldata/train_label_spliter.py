import pandas as pd
import fileinput
import subprocess
import numpy as np
from pandas.core.arrays.sparse import dtype

'''
author          : jinseo Choi
param           : training data path(,csv), test data path(,csv), 
                  path to save processd model_train data, label data 
                  path to save processd model_test data, label data                   
discription     : The module for split MNIST Training, Test data to train, label without header.
                  cause pandas DataFrame takes too many memory. 
                  so, I need to load csv and convert numpy array directly to split train, label column as file, 
                  then, load each file without Pandas Dataframe that save some memory. 
lastest revised : 03.09.2021 
conatact        : tjwjs513@gmail.com 
'''
def MNIST_remove_header(train_data_path, test_data_path, x_train_path, y_train_path, x_test_path, y_test_path):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    print("Processing...\n")
    x_train = df_train.drop(['label'],axis=1)
    #x_train = x_train.to_numpy()
    x_train = pd.DataFrame(x_train,dtype=int)
    x_train.to_csv(x_train_path,header = None, index=None)
    y_train = df_train['label']
    y_train = pd.DataFrame(y_train, dtype=np.int)
    y_train.to_csv(y_train_path,header = None, index=None)        
    x_test = df_test.drop(['label'], axis = 1)
    #x_test = x_test.to_numpy()
    x_test = pd.DataFrame(x_test, dtype=np.int)
    x_test.to_csv(x_test_path,header = None, index=None)        
    y_test = df_test['label']
    y_test = pd.DataFrame(y_test, dtype=np.int)
    y_test.to_csv(y_test_path, header = None, index=None)
    print("Done")
    
    
#MNIST_remove_header('/home/js/Mnist/integrated_train.csv','/home/js/Mnist/integrated_test.csv',
#                    '/home/js/Mnist/integrated_xtrain.csv','/home/js/Mnist/integrated_ytrain.csv',
#                    '/home/js/Mnist/integrated_xtest.csv','/home/js/Mnist/integrated_ytest.csv')
