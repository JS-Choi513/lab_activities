
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.python.ops.gen_math_ops import Max, mul
device_lib.list_local_devices()
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import glob 
import random
import threadpool
import copy


time0 = time.time() 
df_train = pd.read_csv("/home/js/train_hhdata.csv")
df_test = pd.read_csv("/home/js/test_hhdata.csv")
y_train = df_train['label']
x_train = df_train.drop(['label'],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = df_test['label']
x_test = df_test.drop(['label'], axis=1)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


print(x_train.dtype)
random.seed(100)

shuffled_train_result = np.zeros((1026,90000,1))
shuffled_label_result = np.zeros((1026,))

#gpus = tf.config.experimental.list_physical_devices('GPU')
'''

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#동적변수 생성 (안씀)
def generate_parted_array(arr_num, divide_range):
      for i in range(arr_num):
            globals()['Thread{}_train'.format(i)] = np.zeros((divide_range,784,1))
            globals()['Thread{}_label'.format(i)] = np.zeros((divide_range,))
'''            
def multi_shuffling(train_arr,label_arr,threadnum):
      time1 = time.time()
      rowcount = len(np.arange(train_arr.shape[0]))  
      global divide_range
      divide_range = int(rowcount/threadnum)
      global train
      train = train_arr
      global label
      label = label_arr
      randomidx = random.sample(range(threadnum),threadnum)
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            offset_idx = 0
            for exec_threadnum in range(threadnum):
                  start_idx = offset_idx*divide_range
                  offset_idx +=1
                  end_idx = offset_idx*divide_range
                  executor.submit(part_shuffle,start_idx, end_idx, randomidx[exec_threadnum])                
      img = shuffled_train_result.reshape(1026,300,300,1)
      lab = shuffled_label_result
      print("shuffling time",time.time()-time1)
      return img, lab
    

def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      start_point = divide_range*random_idx
      end_point = start_point+divide_range
      tmp_x = train[start_point:end_point]
      tmp_y = label[start_point:end_point]      
      shuffled_train_result[start: end] = tmp_x[suffleidx]
      shuffled_label_result[start: end] = tmp_y[suffleidx]
      

batching_testimg = x_test.reshape(255,300,300,1)
batching_testlab = y_test.reshape(-1,)
print(batching_testimg.shape)
print(batching_testlab.shape)
'''tf.data'''
# multishuffle + tf.data
#train_ds = tf.data.Dataset.from_tensor_slices((batch_train_x, batch_train_y)).batch(32)
#train_ds = tf.data.Dataset.from_tensor_slices(multi_shuffling(x_train,y_train,400)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(32)
'''tf.data'''


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16, (3,3), activation='relu')
        self.mxpool = MaxPool2D(2,2)
        self.flatten = Flatten()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.mxpool(x)
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
  train_ds = tf.data.Dataset.from_tensor_slices(multi_shuffling(x_train,y_train,800)).batch(32)
  for images, labels in train_ds:
        train_step(images, labels)

  for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))                         
  

print("Execution time") 
print(time.time() - time0)  

