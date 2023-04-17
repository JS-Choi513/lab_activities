
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
from tensorflow.python.ops.nn_impl import log_poisson_loss
device_lib.list_local_devices()
import threading
from concurrent.futures import wait, ALL_COMPLETED, ProcessPoolExecutor
import concurrent.futures
import glob 
import random

import threading 
import copy

df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
THREAD_NUM = 24




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
train_range_pairs_per_thread = []
test_range_pairs_per_thread = []
train_rowcount = len(np.arange(x_train.shape[0]))  
test_rowcount = len(np.arange(x_test.shape[0]))
train_divide_range = int(train_rowcount/THREAD_NUM)
test_divide_range = int(test_rowcount/THREAD_NUM)
remain_test_data = test_rowcount % THREAD_NUM
randomidx = random.sample(range(THREAD_NUM),THREAD_NUM)
train_offset_idx = 0
test_offset_idx = 0
test_accuracy_arr = []
test_loss_arr = []



for i in range(THREAD_NUM):
      train_start_idx = train_offset_idx*train_divide_range
      test_start_idx = test_offset_idx*train_divide_range      
      train_offset_idx +=1
      test_offset_idx +=1
      train_end_idx = train_offset_idx*train_divide_range
      if i == THREAD_NUM-1 and remain_test_data != 0:
            test_end_idx = test_offset_idx*test_divide_range + remain_test_data
      else :
            test_end_idx = test_offset_idx*test_divide_range
      train_range_pairs_per_thread.append((train_start_idx, train_end_idx, randomidx[i]))
      #print(train_range_pairs_per_thread[i][0])
      #print(train_range_pairs_per_thread[i][1])
      #print(train_range_pairs_per_thread[i][2])
      #print("\n")
      test_range_pairs_per_thread.append(randomidx[i])


def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(train_divide_range)
      #np.random.shuffle(suffleidx)    
      start_point = train_divide_range*random_idx
      end_point = start_point+train_divide_range
      tmp_x = x_train[start_point:end_point]
      tmp_y = y_train[start_point:end_point]   
      shuffled_train_result[start:end] = tmp_x[suffleidx]
      shuffled_label_result[start:end] = tmp_y[suffleidx]
      a = shuffled_train_result[start: end].reshape(-1,28,28,1)
      b = shuffled_label_result[start: end].reshape(-1,)
      return a,b

def part_test(random_idx):      
      start_point = test_divide_range*random_idx
      end_point = start_point+test_divide_range
      test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg[start_point:end_point],batching_testlab[start_point:end_point])).batch(512)
      for test_images, test_labels in test_ds:
            lock.acquire()
            test_step(test_images, test_labels)
            lock.release()
      test_loss_arr.append(test_loss.result())
      test_accuracy_arr.append(test_accuracy.result()*100)




def multi_epoch(threadnum):
      randomidx = random.sample(range(threadnum),threadnum)
      futures = []
      with ProcessPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(part_epoch, train_range_pairs_per_thread[randomidx[exec_threadnum]][0], 
                                                             train_range_pairs_per_thread[randomidx[exec_threadnum]][1],
                                                             train_range_pairs_per_thread[randomidx[exec_threadnum]][2]))                         
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      futures = []
      with ProcessPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(part_test, test_range_pairs_per_thread[exec_threadnum]))
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
            #executor.shutdown(wait=True)            

def part_epoch(start, end, random_idx):
      train_d = tf.data.Dataset.from_tensor_slices(part_shuffle(start,end, random_idx))
      train_ds = train_d.batch(512)      
      for images, labels in train_ds:
            lock.acquire()
            train_step(images, labels)
            lock.release()
      return True



      
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
      multi_epoch(THREAD_NUM)        
      avg_test_loss =   np.array(test_loss_arr)
      loss = np.mean(avg_test_loss)
      #print(loss)
      avg_test_accuracy =   np.array(test_accuracy_arr)
      accuracy = np.mean(avg_test_accuracy)
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(epoch+1,
                             train_loss.result(),
                             train_accuracy.result()*100,
                             loss,
                             accuracy)) 
 


print("Execution time") 
print(time.time() - time0)   
      


