
# -*- coding: utf-8 -*-
import time
import datetime
from pandas.core.indexes.period import period_range
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
import queue
tf.config.run_functions_eagerly(True)
df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
THREAD_NUM = 6

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)



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

def get_part_data(start, end):
      train = x_train[start:end].reshape(-1,28,28,1)
      lab = y_train[start:end].reshape(-1,)
      return train, lab
      

test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(128,num_parallel_calls=1)

def multi_epoch(nth_epoch, threadnum, train_arr,label_arr):
      futures = []
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(part_epoch, range_pairs_per_thread[exec_threadnum][0], range_pairs_per_thread[exec_threadnum][1], range_pairs_per_thread[exec_threadnum][2]))                         
      #wait(futures,timeout=None,return_when=ALL_COMPLETED)
      
            executor.shutdown(wait=True)            
      for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(nth_epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100)) 



def part_epoch(start, end, random_idx):
      data = get_part_data(start,end)
      train = data[0]
      label = data[1]
      print(type(train))

      print(type(label))
      a = pd.DataFrame(train)
      a.to_csv("a.csv")
      b = pd.DataFrame(label)
      b.to_csv("b.csv")
      train_d = tf.data.Dataset.from_tensor_slices((train, label))
      train_ds = train_d.batch(32)      
      for images, labels in train_ds:
            #print("moddel"),
            train_step(images, labels)
            #lock.release()
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
            print("prediction", predictions)
            loss = loss_object(labels, predictions)
            print("Loss",loss)
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
      


