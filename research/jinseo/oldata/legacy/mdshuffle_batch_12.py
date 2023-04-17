
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
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
THREAD_NUM = 12




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
for i in range(THREAD_NUM):
      start_idx = offset_idx*divide_range
      offset_idx +=1
      end_idx = offset_idx*divide_range
      range_pairs_per_thread.append((start_idx, end_idx, randomidx[i]))



def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      print("shuffle")
      start_point = divide_range*random_idx
      print("shuffle2")
      end_point = start_point+divide_range
      print("pointing")
      tmp_x = x_train[start_point:end_point]
      print("pointing2")
      tmp_y = y_train[start_point:end_point]      
      print("trainsport")
      shuffled_train_result[start: end] = tmp_x[suffleidx]
      print("trainsport2")
      shuffled_label_result[start: end] = tmp_y[suffleidx]
      print("trainsport3")
      a = shuffled_train_result[start: end].reshape(-1,28,28,1)
      print("trainsport3")
      b = shuffled_label_result[start: end].reshape(-1,)
      print("trainsport3")
      return a,b


test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(256,num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
                  futures.append(executor.submit(part_epoch, range_pairs_per_thread[exec_threadnum][0], range_pairs_per_thread[exec_threadnum][1], range_pairs_per_thread[exec_threadnum][2]))                         
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      
            #executor.shutdown(wait=True)            
      for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(nth_epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100)) 



def part_epoch(start, end, random_idx):
      
      train_d = tf.data.Dataset.from_tensor_slices(part_shuffle(start,end, random_idx))
      train_ds = train_d.batch(256,num_parallel_calls=tf.data.experimental.AUTOTUNE)      
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
      multi_epoch(epoch,THREAD_NUM,x_train,y_train)           


print("Execution time") 
print(time.time() - time0)   
      


