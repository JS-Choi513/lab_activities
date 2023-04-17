
# -*- coding: utf-8 -*-
# MNIST 데이터를 다운로드 한다.
#thread = 100
#thread 1 
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
# manual input pipeline#######################3 
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.python.ops.gen_math_ops import mul
device_lib.list_local_devices()
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import glob 
import random
import threadpool
import copy
from collections import deque

filename = '/media/js/test/cpu.txt'

def logWrite(data):    
  f = open(filename,"a")
  f.write(data)
  f.close()

df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
#df_train = pd.read_csv("/home/js/Mnist/mnist_train1.csv")
#df_test = pd.read_csv("/home/js/Mnist/mnist_test1.csv")
y_train = df_train['label']
x_train = df_train.drop(['label'],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = df_test['label']
x_test = df_test.drop(['label'], axis=1)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


shuffled_train_result = np.zeros((540000,784,1))
shuffled_label_result = np.zeros((540000,))    

tmp2_x = np.zeros((45000,784,1))
tmp2_y = np.zeros((45000,))  

tmp2_x1 = np.zeros((45000,784,1))
tmp2_y1 = np.zeros((45000,))  

tmp2_x2 = np.zeros((45000,784,1))
tmp2_y2 = np.zeros((45000,))  

tmp2_x3 = np.zeros((45000,784,1))
tmp2_y3 = np.zeros((45000,))  

tmp2_x4 = np.zeros((45000,784,1))
tmp2_y4 = np.zeros((45000,))  

tmp2_x5 = np.zeros((45000,784,1))
tmp2_y5 = np.zeros((45000,))  

tmp2_x6 = np.zeros((45000,784,1))
tmp2_y6 = np.zeros((45000,))  

tmp2_x7 = np.zeros((45000,784,1))
tmp2_y7 = np.zeros((45000,))  

tmp2_x8 = np.zeros((45000,784,1))
tmp2_y8 = np.zeros((45000,))  

tmp2_x9 = np.zeros((45000,784,1))
tmp2_y9 = np.zeros((45000,))  

tmp2_x10 = np.zeros((45000,784,1))
tmp2_y10 = np.zeros((45000,))  

tmp2_x11 = np.zeros((45000,784,1))
tmp2_y11 = np.zeros((45000,))  

 
random.seed(1000)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#time0 = time.time() 

def multi_shuffling(train_arr,label_arr,threadnum, seed_tt):
      time0 = time.time() 
      #logWrite("shuffle called")
      #randomidx_cnt = 0


 


      rowcount = len(np.arange(train_arr.shape[0]))   
      global divide_range
      divide_range = int(rowcount/threadnum)
      pool = ThreadPoolExecutor(threadnum)
      
      
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
        exec1 = 0        
        start = exec1 * divide_range
        end = (exec1 + 1) * divide_range
        executor.submit(part_shuffle,start, end)

        exec1 += 1
        start = exec1 * divide_range
        end = (exec1 + 1) * divide_range
        executor.submit(part_shuffle1,start, end)

        exec1 += 1
        start = exec1 * divide_range
        end = (exec1 + 1) * divide_range
        executor.submit(part_shuffle2,start, end)

        exec1 += 1
        start = exec1 * divide_range
        end = (exec1 + 1) * divide_range
        executor.submit(part_shuffle3,start, end)
        


      print("shuffle time",time.time()-time0 )

def part_shuffle(start, end):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      tmp_x = x_train[start:end]
      tmp_y = y_train[start:end]
      tmp2_x[0:] = tmp_x[suffleidx]
      tmp2_y[0:] = tmp_y[suffleidx]

def part_shuffle1(start, end):    
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)     
      tmp_x = x_train[start:end]
      tmp_y = y_train[start:end]
      tmp2_x1[0:] = tmp_x[suffleidx]
      tmp2_y1[0:] = tmp_y[suffleidx]
      
def part_shuffle2(start, end):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)     
      tmp_x = x_train[start:end]
      tmp_y = y_train[start:end]
      tmp2_x2[0:] = tmp_x[suffleidx]
      tmp2_y2[0:] = tmp_y[suffleidx]

def part_shuffle3(start, end):
      print(divide_range)
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)     
      tmp_x = x_train[start:end]
      tmp_y = y_train[start:end]
      tmp2_x3[0:] = tmp_x[suffleidx]
      tmp2_y3[0:] = tmp_y[suffleidx]





time1 = time.time()         
#print("x_train_shape",x_train.shape)
#print("y_train_shape",y_train.shape)

'''Numpy shuffling'''
#np.random.seed(1000)
#s = np.arange(x_train.shape[0])
#np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
#train_dx = x_train[s]
#train_dy = y_train[s]
'''Numpy shuffling'''

multi_shuffling(x_train,y_train,12,1000)

exec2 = 0        
start = exec2 * divide_range
end = (exec2 + 1) * divide_range
print(divide_range)
shuffled_train_result[start: end] = tmp2_x[0:divide_range]#[0:]#.copy()
shuffled_label_result[start: end] = tmp2_y[0:divide_range]#[0:]#.copy()

exec2 +=1
start = exec2 * divide_range
end = (exec2 + 1) * divide_range
shuffled_train_result[start: end] = tmp2_x1[0:divide_range]#[0:]#.copy()
shuffled_label_result[start: end] = tmp2_y1[0:divide_range]#[0:]#.copy()



exec2 +=1       
start = exec2 * divide_range
end = (exec2 + 1) * divide_range
shuffled_train_result[start: end] = tmp2_x2[0:divide_range]#[0:]#.copy()
shuffled_label_result[start: end] = tmp2_y2[0:divide_range]#[0:]#.copy()



exec2 +=1     
start = exec2 * divide_range
end = (exec2 + 1) * divide_range
shuffled_train_result[start: end] = tmp2_x3[0:divide_range]#[0:]#.copy()
shuffled_label_result[start: end] = tmp2_y3[0:divide_range]#[0:]#.copy()


print(time.time() - time1)  
#batching_img = train_dx.reshape(-1,25,28,28,1)
#batching_lab = train_dy.reshape(-1,25)
batching_img = shuffled_train_result.reshape(-1,25,28,28,1)
batching_lab = shuffled_label_result.reshape(-1,25)
print(batching_img.shape)
print(batching_lab.shape)
batching_testimg = x_test.reshape(-1,25,28,28,1)
batching_testlab = y_test.reshape(-1,25)

'''tf.data'''
#train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32, num_parallel_calls=10)
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32,  num_parallel_calls=10)
'''tf.data'''


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 10

for epoch in range(EPOCHS):
  for images, labels in zip(batching_img, batching_lab):
    train_step(images, labels)
    '''tf.data 사용 시 주석해제 '''
#  for images, labels in train_ds:
#        #print(train_ds)
#    train_step(images, labels)

  for test_images, test_labels in zip(batching_testimg,batching_testlab):
    test_step(test_images, test_labels)
    '''tf.data 사용 시 주석해제 '''
#  for test_images, test_labels in test_ds:
#    test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
print("Execution time") 
print(time.time() - time0)  

