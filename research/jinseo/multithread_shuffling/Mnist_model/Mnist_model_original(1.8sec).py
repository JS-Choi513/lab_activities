
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
'''
      seed = seed # 랜덤셔플 시드값
      random.seed(seed)
      randomidx = random.sample(range(threadnum),threadnum)
      train_arr = train_arr #셔플할 훈련데이터 540000,784,1
      label_arr = label_arr #셔플할 라벨데이터 540000,1
      rowcount = len(np.arange(train_arr.shape[0]))# 입력행렬 라인개수 
      divide_range = int(rowcount/threadnum) # 전체데이터를 행단위로 스레드마다 일정량 할당되도록 분배 
      idx_chk = 0
      randomidx_cnt = 0
      shuffle_idx_range = []
      shuffle_idx_range_cnt = 0
      global shuffled_train_result 
      shuffled_train_result = np.zeros((540000,784,1))
      global shuffled_label_result
      shuffled_label_result = np.zeros((540000,))
      pool = ThreadPoolExecutor(threadnum)
      with pool as executer:
        for devideidx in range(threadnum): # 0 ~ threadnum
              if randomidx[randomidx_cnt] != threadnum-1: # 스레드개수만큼 데이터가 나누어떨어지지 않을 때 마지막 쓰레드가 나머지까지 들고감 
                parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
                parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
              else:
                    parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:]
                    parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:]
              future = executer.submit(part_shuffle, parted_train_arr, parted_label_arr,
                                       idx_chk, divide_range, shuffle_idx_range, seed)                  
              shuffle_idx_range_cnt+=1
              randomidx_cnt+=1              
              idx_chk += divide_range        
      global batching_img 
      batching_img = shuffled_train_result.reshape(-1,25,28,28,1)
      global batching_lab 
      batching_lab = shuffled_label_result.reshape(-1,25)            
    '''

#time0 = time.time() 

def multi_shuffling(train_arr,label_arr,threadnum, seed_tt):
      time0 = time.time() 
      #logWrite("shuffle called")
      #randomidx_cnt = 0
      global shuffled_train_result 
      shuffled_train_result = np.zeros((540000,784,1))
      global shuffled_label_result
      shuffled_label_result = np.zeros((540000,))    
      random.seed(seed_tt)
      randomidx = random.sample(range(threadnum),threadnum)# 인덱스 셔플 
      rowcount = len(np.arange(train_arr.shape[0]))
      global divide_range
      divide_range = int(rowcount/threadnum)
      range_list=[0 for i in range(threadnum)]      #0n ~ 24n
      offset_list=[0 for i in range(threadnum)]
      non_rnd_range = [0 for i in range(threadnum)]      
      non_rnd_offset = [0 for i in range(threadnum)]     
      range1 = 0
      for i in range(threadnum):          
          offset_list[randomidx[i]] = range1 # 0, n, 2n, 3n, 4n, ... 23n #
          non_rnd_offset[i] = range1
          range1+=divide_range  
          range_list[randomidx[i]] = range1  # n, 2n, 3n, 4n, 5n, ....24
          non_rnd_range[i] = range1      
      pool = ThreadPoolExecutor(threadnum)
      
      
      with concurrent.futures.ThreadPoolExecutor(max_workers=threadnum) as executor:
        #jobs = []
        time0 = time.time()  
        for exec1 in range(threadnum):
              
              start = exec1 * divide_range
              end = (exec1 + 1) * divide_range
              job = executor.submit(part_shuffle,start, end, non_rnd_offset[exec1],non_rnd_range[exec1],exec1, seed_tt)
              #jobs.append(job)
                            
      #concurrent.futures.wait(jobs,timeout=None) 
      print("shuffle time",time.time()-time0 )
def part_shuffle(start, end, non_rnd_offset, non_rnd_range,exec1, seed_tmp):
      #time0 = time.time() 

      random.seed(1000)
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)     
      tmp_x = x_train[start:end]
      tmp_y = y_train[start:end]
      
      #print("shuffle time",time.time()-time0 )
      
      #time1 = time.time() 
      shuffled_train_result[non_rnd_offset: non_rnd_range] = tmp_x[suffleidx]#[0:]#.copy()
      shuffled_label_result[non_rnd_offset: non_rnd_range] = tmp_y[suffleidx]#[0:]#.copy()
      #print("assign time",time.time()-time1 )

'''파라미터에 스레드 개수만 입력하면 내부적으로 반영되도록 변경함'''
'''
def multi_shuffling(train_arr,label_arr,threadnum, seed_tt):
      randomidx_cnt = 0
      global shuffled_train_result 
      shuffled_train_result = np.zeros((270000,784,1))
      global shuffled_label_result
      shuffled_label_result = np.zeros((270000,))    
      random.seed(seed_tt)
      randomidx = random.sample(range(threadnum),threadnum)# 인덱스 셔플 
      rowcount = len(np.arange(train_arr.shape[0]))
      divide_range = int(rowcount/threadnum)
      idx_chk = 0
      randomidx_cnt = 0
      shuffle_idx_range = []
      shuffle_idx_range_cnt = 0
      #range_list=[0 for i in range(threadnum)]      #0n ~ 24n
      #offset_list=[0 for i in range(threadnum)]
      #non_rnd_range = [0 for i in range(threadnum)]      
      #non_rnd_offset = [0 for i in range(threadnum)]     
      #range1 = 0
      #for i in range(threadnum):          
      #    offset_list[randomidx[i]] = range1 # 0, n, 2n, 3n, 4n, ... 23n #
      #    non_rnd_offset[i] = range1
      #    range1+=divide_range  
      #    range_list[randomidx[i]] = range1  # n, 2n, 3n, 4n, 5n, ....24
      #    non_rnd_range[i] = range1      

      pool = ThreadPoolExecutor(threadnum)
      with concurrent.futures.ThreadPoolExecutor(max_workers=threadnum) as executor:
        jobs = []
        for exec1 in range(threadnum):
              if randomidx[randomidx_cnt] != threadnum-1: # 스레드개수만큼 데이터가 나누어떨어지지 않을 때 마지막 쓰레드가 나머지까지 들고감 
                parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
                parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:randomidx[randomidx_cnt]*divide_range+divide_range]
              else:
                    parted_train_arr = train_arr[randomidx[randomidx_cnt]*divide_range:]
                    parted_label_arr = label_arr[randomidx[randomidx_cnt]*divide_range:]
              job = executor.submit(part_shuffle,parted_train_arr, parted_label_arr,  idx_chk, divide_range, shuffle_idx_range, seed_tt)  
              jobs.append(job)                  
              shuffle_idx_range_cnt+=1
              randomidx_cnt+=1              
              idx_chk += divide_range    
              #if randomidx[randomidx_cnt] != threadnum-1: # 스레드개수만큼 데이터가 나누어떨어지지 않을 때 마지막 쓰레드가 나머지까지 들고감 
              #  parted_train_arr = train_arr[offset_list[exec1]:range_list[exec1]]
              #  parted_label_arr = label_arr[offset_list[exec1]:range_list[exec1]]                
              #else:
              #      parted_train_arr = train_arr[offset_list[exec1]:]
              #      parted_label_arr = label_arr[offset_list[exec1]:]

              
      concurrent.futures.wait(jobs,timeout=None) 

def part_shuffle(parted_train_arr, parted_label_arr, offset, range, shuffle_idx_range, seed_tt):
      random.seed(seed_tt)
      suffleidx = np.arange(parted_train_arr.shape[0])
      np.random.shuffle(suffleidx)     
      shuffled_train_arr = parted_train_arr[suffleidx]
      shuffled_label_arr = parted_label_arr[suffleidx]
      shuffled_train_result[offset:range] = shuffled_train_arr[0:].copy()
      shuffled_label_result[offset:range] = shuffled_label_arr[0:].copy()
'''    
      




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

multi_shuffling(x_train,y_train,100,1000)
print(time.time() - time1)  
#batching_img = train_dx.reshape(-1,25,28,28,1)
#batching_lab = train_dy.reshape(-1,25)
#batching_img = shuffled_train_result.reshape(-1,25,28,28,1)
#batching_lab = shuffled_label_result.reshape(-1,25)
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

