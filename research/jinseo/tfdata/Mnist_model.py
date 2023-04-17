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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#      
time0 = time.time() 

def multi_shuffling(train_arr,label_arr,threadnum, seed):
      seed = seed # 랜덤셔플 시드값
      random.seed(seed)
      randomidx = random.sample(range(threadnum),threadnum)
      train_arr = train_arr #셔플할 훈련데이터 540000,784,1
      label_arr = label_arr #셔플할 라벨데이터 540000,1
      rowcount = len(np.arange(train_arr.shape[0]))# 입력행렬 라인개수 
      divide_range = int(rowcount/threadnum) # 전체데이터를 행단위로 스레드마다 일정량 할당되도록 분배 
      idx_chk = 0
      thread_cnt = 0
      randomidx_cnt = 0
      shuffled_array = []
      global shuffled_train_result 
      shuffled_train_result = np.zeros((540000,784,1))
      global shuffled_label_result
      shuffled_label_result = np.zeros((540000,1))
      pool = ThreadPoolExecutor(threadnum)
      with pool as executer:
        for devideidx in range(threadnum): # 0 ~ threadnum
              if randomidx[randomidx_cnt] != threadnum-1: # 스레드개수만큼 데이터가 나누어떨어지지 않을 때 마지막 쓰레드가 나머지까지 들고감 
                parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
                parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
                #parted_train_arr = train_arr[idx_chk:divide_range+idx_chk]
                #parted_label_arr = label_arr[idx_chk:divide_range+idx_chk]
              else:
                    parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:]
                    parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:]
              future = executer.submit(part_shuffle, parted_train_arr, parted_label_arr, idx_chk, divide_range, seed)              
              shuffled_array.append(future)              
              randomidx_cnt+=1              
              idx_chk += divide_range                                  
        
      #return train, label#shuffled_train_result, shuffled_label_result#

def part_shuffle(parted_train_arr, parted_label_arr, offset, range, seed):
      np.random.seed(seed)#
      suffleidx = np.arange(parted_train_arr.shape[0])#45000
      #print("parted_train_arr[0]",parted_train_arr.shape)# 45000, 784, 1
      np.random.shuffle(suffleidx)
      shuffled_train_arr = parted_train_arr[suffleidx]
      shuffled_label_arr = parted_label_arr[suffleidx]
      shuffled_train_result[offset:range] = shuffled_train_arr
      shuffled_train_result[offset:range] = shuffled_label_arr
      #return shuffled_train_arr, shuffled_label_arr

df_train = pd.read_csv("/home/js/Mnist/integrated_train500.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test500.csv")

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

#x_reshp = x_train.reshape(-1,32,28,28,1)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]



time1 = time.time()         
print("x_train_shape",x_train.shape)
print("y_train_shape",y_train.shape)


np.random.seed(1000)
s = np.arange(x_train.shape[0])
np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
train_dx = x_train[s]
train_dy = y_train[s]



#multi_shuffling(x_train,y_train,1,1000)

#nptrain = np.array(train)Ram penisRam penis
#nplabel = np.array(label)
#print("multi_shuffling_result train: ",nptrain)
#print("multi_shuffling_result train shape: ",nptrain.shape)
#print("multi_shuffling_result label: ",nplabel)
#print("multi_shuffling_result label shape: ",nplabel.shape)
#(2400, 25, 28, 28, 1)
#(2400, 25)

print(time.time() - time1)  

#셔플링된 데이터 

batching_img = train_dx.reshape(-1,25,28,28,1)
batching_lab = train_dy.reshape(-1,25)
#batching_img = shuffled_train_result.reshape(-1,25,28,28,1)
#batching_lab = shuffled_label_result.reshape(-1,25)
#print(batching_img.shape)
#print(batching_lab.shape)
batching_testimg = x_test.reshape(-1,25,28,28,1)
batching_testlab = y_test.reshape(-1,25)

#print("pipline shuffing +batching") 
#print(time.time() - time1)  
#train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32, num_parallel_calls=10)
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32,  num_parallel_calls=10)


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
    #print(train_ds)
    train_step(images, labels)
#  for images, labels in train_ds:
#        #print(train_ds)
#    train_step(images, labels)

  for test_images, test_labels in zip(batching_testimg,batching_testlab):
    test_step(test_images, test_labels)

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

