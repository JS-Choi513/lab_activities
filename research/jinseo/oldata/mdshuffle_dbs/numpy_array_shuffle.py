import numpy as np
import Data_loader
import threading
import tensorflow as tf
import pandas as pd
x_train = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_xtrain.csv')
y_train = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_ytrain.csv')
x_test = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_xtest.csv')
y_test = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_ytest.csv')
lock = threading.Lock()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train = x_train[..., tf.newaxis]
#x_test = x_test[..., tf.newaxis]

shuffled_train_result = np.zeros((540000,784,1))
shuffled_label_result = np.zeros((540000,))
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)
print("xtrain shape\n",x_train.shape)
print("ytrain shape\n",y_train.shape)
print("xtest shape\n",x_test.shape)
print("ytest shape\n",y_test.shape)



#arr = np.array(range(100))
#print(arr)
#np.random.shuffle(arr)
#print(arr)

a = pd.DataFrame(x_train)
print(a.info())
print(a.head(5))
np.random.shuffle(x_train)
b = pd.DataFrame(x_train)
print(b.info())
print(b.head(5))