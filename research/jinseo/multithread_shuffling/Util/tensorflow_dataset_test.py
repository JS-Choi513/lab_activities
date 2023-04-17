import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
from concurrent.futures import ThreadPoolExecutor
import sys
import copy
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds

# 4D to 2D dataFrame Generator
def dataFrame_Generator(numpy_arr,column):
    threeDshape = numpy_arr.shape[:-1]
    print("method",threeDshape)
    matrix_prod = np.prod(threeDshape)
    print("method producy",matrix_prod)
    twoDshape = numpy_arr.reshape(matrix_prod,numpy_arr.shape[-1])
    print("method",twoDshape)
    dataFrame = pd.DataFrame(twoDshape,columns=[column,"0","1"])
    return dataFrame

def dataAgumentater(src_path, dest_path, aug_count):
    column_name=1
    for i in range(aug_count):
        line_cnt=0
        f1 = open(src_path,'r')
        f2 = open(dest_path,'a')
        line = f1.readline()
        if column_name == 1:
            f2.write(line)
            column_name = 0
            line_cnt+=1
        while True:
            line = f1.readline()
            if not line: break
            f2.writelines(line)            
        print("line count: ",line_cnt)    
        line_cnt+=1
        f1.close()
        f2.close()
    result = pd.read_csv(dest_path)        
    print(result)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data(label_mode="fine")
assert x_train.shape == (50000,32,32,3)
assert x_test.shape == (10000,32,32,3)
assert y_train.shape == (50000,1)
assert y_test.shape == (10000,1)
threedshape = x_train.shape[:-1]
prodx= np.prod(threedshape)
print("x_trina shape",x_train.shape)
print("y_label shape",y_train.shape)
print("x_test shape",x_test.shape)
print("y_testlabel shape",y_test.shape)

twodshape= x_train.reshape(prodx,x_train.shape[-1])# 얘를 데이터 프레임으로 넣어서 
#train_label = pd.read_csv("/home/js/train_label.csv",index_col=0)
#test_label = pd.read_csv("/home/js/test_label.csv",index_col=0)
'''
xtrain = pd.DataFrame(twodshape,columns=['train',"0","1"])
#xtrain.to_csv("/home/js/xtrain.csv",index=False)
xtest = dataFrame_Generator(x_test, 'train')
#xtest.to_csv("/home/js/xtest.csv",index=False)


ytrain = pd.DataFrame(y_train,columns=['label'])
ytest = pd.DataFrame(y_test,columns=['label'])
train = pd.concat([xtrain,ytrain],axis=1)
test = pd.concat([xtest,ytest],axis=1)
print("train")
print(train)
print("test")
print(test)

print(train)
train.to_csv("/home/js/train.csv",index=False)
test.to_csv("/home/js/test.csv",index=False)

#train_label = pd.read_csv("/home/js/train_label.csv",index_col=0)
#test_label = pd.read_csv("/home/js/test_label.csv",index_col=0)

'''

train_label = pd.DataFrame(y_train,columns=['label'])
test_label = pd.DataFrame(y_test,columns=['label'])
train_label.to_csv("/home/js/train_label.csv",index=False)
test_label.to_csv("/home/js/test_label.csv",index=False)
dataAgumentater("/home/js/train_label.csv","/home/js/2train_label.csv",2)
dataAgumentater("/home/js/test_label.csv","/home/js/2test_label.csv",2)