
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.framework.func_graph import ALLOWLIST_COLLECTIONS
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.python.ops.gen_math_ops import mul
device_lib.list_local_devices()
import threading
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
import concurrent.futures
import glob 
import random
import threadpool
import threading 
import copy

df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
THREAD_NUM = 6

#1


time0 = time.time() 
y_train = df_train['label']
x_train = df_train.drop(['label'],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = df_test['label']
x_test = df_test.drop(['label'], axis=1)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
random.seed(1000)

shuffled_train_result = np.zeros((540000,784,1))
shuffled_label_result = np.zeros((540000,))
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)

lock = threading.Lock()
range_pairs_per_thread = []
rowcount = len(np.arange(x_train.shape[0]))  
divide_range = int(rowcount/THREAD_NUM)
randomidx = random.sample(range(THREAD_NUM),THREAD_NUM)
offset_idx = 0

class MyModel(Model): #keras의 모델을 상속받음 
      def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10, activation='softmax')
            
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam()
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            
      
      def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

      def get_loss(self, label, predictions):
            return self.loss_object(label, predictions)


      @tf.function
      def train_step(self, images, labels):
            with tf.GradientTape() as tape:
            # print("init train_step")
                  predictions = self.call(images) # 쓰레드마다 할당된 모델 객체에 대해 예측 
                  #print("Pridiction_complete",predictions)
                  loss = self.loss_object(labels, predictions)
                  #print("loss", loss)
            # print("loss complete")
            gradients = tape.gradient(loss, self.trainable_variables)
            #global_gradients.append(gradient)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)


      @tf.function
      def test_step(self, images, labels):
            predictions = self.call(images)
            t_loss = self.loss_object(labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

model_arr = []
global_gradients = []
for i in range(THREAD_NUM):
      model_arr.append(MyModel())
      print("add model\n")

for i in range(THREAD_NUM):
      start_idx = offset_idx*divide_range
      offset_idx +=1
      end_idx = offset_idx*divide_range
      range_pairs_per_thread.append((start_idx, end_idx, randomidx[i]))



def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      start_point = divide_range*random_idx
      end_point = start_point+divide_range
      tmp_x = x_train[start_point:end_point]
      tmp_y = y_train[start_point:end_point]      
      shuffled_train_result[start: end] = tmp_x[suffleidx]
      shuffled_label_result[start: end] = tmp_y[suffleidx]
      a = shuffled_train_result[start: end].reshape(-1,28,28,1)
      b = shuffled_label_result[start: end].reshape(-1,)
      return a,b


test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(128,num_parallel_calls=1)

def multi_epoch(nth_epoch, threadnum, train_arr,label_arr):
      #rowcount = len(np.arange(train_arr.shape[0]))  
      #global divide_range
      #divide_range = int(rowcount/threadnum)
      #randomidx = random.sample(range(threadnum),threadnum)
      futures = []
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
       #     offset_idx = 0
            for exec_threadnum in range(threadnum):
        #          start_idx = offset_idx*divide_range
        #          offset_idx +=1
        #          end_idx = offset_idx*divide_range
                  futures.append(executor.submit(part_epoch, range_pairs_per_thread[exec_threadnum][0], range_pairs_per_thread[exec_threadnum][1], range_pairs_per_thread[exec_threadnum][2], exec_threadnum))                         
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      
            #executor.shutdown(wait=True)            




def part_epoch(start, end, random_idx, model_idx):
      
      train_d = tf.data.Dataset.from_tensor_slices(part_shuffle(start,end, random_idx))
      train_ds = train_d.batch(128)      
      for images, labels in train_ds:
            #lock.acquire()
            model_arr[model_idx].train_step(images, labels)
            #lock.release()
      for test_images, test_labels in test_ds:
            model_arr[model_idx].test_step(test_images, test_labels)
      #template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print("train-test done")
      return True




EPOCHS = 10
for epoch in range(EPOCHS):
      time1 = time.time()
      multi_epoch(epoch,THREAD_NUM,x_train,y_train)           
      print("1epoch", time.time()-time1)


print("Execution time") 
print(time.time() - time0)   
      


