
# -*- coding: utf-8 -*-
import time
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
import threading 
from queue import Queue
from numpy import genfromtxt
import tensorflow as tf
import MNIST_model as MNIST
import Data_loader
import copy
import random
THREAD_NUM = 1
EPOCHS = 10
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)

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
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

shuffled_train_result = np.zeros((540000,784,1))
shuffled_label_result = np.zeros((540000,))
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)
print("xtrain shape\n",x_train.shape)
print("ytrain shape\n",y_train.shape)
print("xtest shape\n",x_test.shape)
print("ytest shape\n",y_test.shape)
random.seed(1000)

time0 = time.time()
# 스레드별 모델 객체 list
train_rowcount = len(np.arange(x_train.shape[0]))  
test_rowcount = len(np.arange(x_test.shape[0]))
randomidx = []
thread_train_flag = []
offset_idx = 0
tf.config.run_functions_eagerly(True)
# 쓰레드별로 분배될 훈련데이터 포지션 저장
parted_train_data_q = Queue(THREAD_NUM) #원본데이터 셔플 후 저장되는 큐
parted_shuffled_data_q = Queue(THREAD_NUM)# 훈련으로 넘어가기 전 대기하는 큐 
train_divide_range = int(train_rowcount/THREAD_NUM)
test_divide_range = int(test_rowcount/THREAD_NUM)
test_accuracy_arr = []
test_loss_arr = []
train_accuracy_arr = [] 
train_loss_arr = []
model_arr = []
BATCH_SIZE = [128,128,128,128,128,128]
THTREAD_TIME = 0
THREAD_STAT = [0,0,0,0,0,0]
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
      def train_step(self, images, labels, model_idx):
            print("%d th thread train func is called"%(model_idx))# %model_idx)
            with tf.GradientTape() as tape:
            # print("init train_step")
                  predictions = self.call(images) # 쓰레드마다 할당된 모델 객체에 대해 예측 
                  #print("Pridiction_complete",predictions)
                  loss = self.loss_object(labels, predictions)
                  #print("loss", loss)
            # print("loss complete")
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)


      @tf.function
      def test_step(self, images, labels, model_idx):
            print("%d th thread test func is called"%(model_idx))# %model_idx)
            predictions = self.call(images)
            t_loss = self.loss_object(labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

def get_random_idx():
      randomidx = random.sample(range(THREAD_NUM),THREAD_NUM)
      return randomidx
      
def get_batch_size(weight)      :
      if weight == 0:
            return BATCH_SIZE[0]
      elif weight == 1:
            return BATCH_SIZE[1]
      elif weight == 2:
            return BATCH_SIZE[2]
      elif weight == 3:
            return BATCH_SIZE[3]
      elif weight == 4:
            return BATCH_SIZE[4]                        
      elif weight == 5:
            return BATCH_SIZE[5]   
      elif weight == 6:
            return BATCH_SIZE[6]

for i in range(THREAD_NUM):
      model_arr.append(MyModel())
      print("add model\n")

      
def enqueue_shuffled_data(randomidx):
      #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      train, label = part_shuffle(randomidx)
      data =(train, label)
      #lock.acquire()
      #print("put part shuffled data!")
      #parted_shuffled_data_q.put_nowait(data)
      parted_shuffled_data_q.put_nowait(data)
      #print("enqueue_shuffle_data complete q size is:",parted_shuffled_data_q.qsize())
      #lock.relase()

def part_shuffle(random_idx):      
      print("called part_shuffle")
      start_point = 0
      end_point = 540000
      print("called part_shuffle2")
      tmp_x = x_train#[start_point:end_point]
      #shuffleidx = np.arange(540000)
      print("called part_shuffle3")
      #np.random.shuffle(shuffleidx)    
      tmp_y = y_train#[start_point:end_point]      
      print("called part_shuffle4")
      shuffled_train_result = tmp_x#[shuffleidx]
      shuffled_label_result = tmp_y#[shuffleidx]
      print("called part_shuffle5")
      a = shuffled_train_result.reshape(-1,28,28,1)
      b = shuffled_label_result.reshape(-1,)
      print("done part shuffle")
      return a, b




def get_part_data(start, end):
      train = x_train[start:end].reshape(-1,28,28,1)
      lab = y_train[start:end].reshape(-1,)
      return train, lab

def data_q_copy():

      if parted_shuffled_data_q.empty() and parted_train_data_q.empty():
            md_shuffle(THREAD_NUM)            
      while parted_shuffled_data_q.empty() is False:            
            job = parted_shuffled_data_q.get()
            data = job
            parted_train_data_q.put(data)
      if parted_train_data_q.empty():
            print("parted_train_data_q is empty after mdshuffle")                              

def md_shuffle(threadnum):
      futures = []
      train_rnd_idx = get_random_idx()
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(enqueue_shuffled_data, train_rnd_idx[exec_threadnum]))     # randon index for merge shuffled data   
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      return True



def multi_epoch(threadnum):
      print("mulit_epoch")
      #if  parted_train_data_q.empty():# 바로 훈련모델에 들어가면 되는 데이터 큐 
      #      print("Ready Queue is empty")
      #      data_q_copy()      
      futures = []
      test_rnd_idx = get_random_idx()
      with ThreadPoolExecutor(max_workers=threadnum) as executor:            
            for exec_threadnum in range(threadnum):                                                                                                                          
                  futures.append(executor.submit(part_epoch,test_rnd_idx[exec_threadnum],exec_threadnum, get_batch_size(exec_threadnum))) #model_index 
            #md_shuffle(THREAD_NUM)
      #wait(futures,timeout=None,return_when=ALL_COMPLETED)
      #md_shuffle(threadnum) # recharge shuffled Queue                              

def part_epoch(random_idx, model_idx, batch_size):
      print("part_epoch")
      #start_time = time.perf_counter()
      #if parted_train_data_q.empty(): 
      #      print(" part_epoch Ready Q is empty") 
      #lock.acquire()
      #data = parted_train_data_q.get_nowait()      
      #print("Getting train data from train parted_train_data_q, remaining data... ",parted_train_data_q.qsize())
      #print("Batch_size: ", batch_size)
      #lock.release()
      train_d = tf.data.Dataset.from_tensor_slices(part_shuffle(randomidx))      
      train_ds = train_d.batch(batch_size)           
      #lock.acquire()
      for images, labels in train_ds:
            model_arr[model_idx].train_step(images, labels, model_idx)
      #lock.release()
      #lock.acquire()            
      train_loss_arr.append(model_arr[model_idx].train_loss.result())          
      #print("batch size %d, %dth thread train is done" % (batch_size, model_idx))
      #print("step train loss",model_arr[model_idx].train_loss.result() )  
      train_accuracy_arr.append(model_arr[model_idx].train_accuracy.result()*100)          
      start_point = test_divide_range*random_idx
      end_point = start_point+test_divide_range
      test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg[start_point:end_point],batching_testlab[start_point:end_point])).batch(32)            
      for test_images, test_labels in test_ds:
            model_arr[model_idx].test_step(test_images, test_labels, model_idx)      
      test_loss_arr.append(model_arr[model_idx].test_loss.result())
      test_accuracy_arr.append(model_arr[model_idx].test_accuracy.result()*100)
      #print("batch size %d, %dth thread test is done" % (batch_size, model_idx))
      #print("step test accuracy",model_arr[model_idx].test_accuracy.result()*100 )
      #lock.release()
      #end_time = time.perf_counter()
      #print("Batch_size :%d , %dth thread test_train Execution time :%d" % (batch_size, model_idx, end_time-start_time))
      #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! thread epoch Done!!!")

      return True



for epoch in range(EPOCHS):
      time1 = time.time()
      multi_epoch(THREAD_NUM)
      avg_test_loss = np.array(test_loss_arr)
      loss = np.mean(avg_test_loss)
      #print("test Loss")
      avg_test_accuracy = np.array(test_accuracy_arr)
      accuracy = np.mean(avg_test_accuracy)
      #print("test Acc")
      avg_train_loss = np.array(train_loss_arr)
      train_loss = np.mean(avg_train_loss)
      #print("train Loss")
      avg_train_accuracy = np.array(train_accuracy_arr)
      train_accuracy = np.mean(avg_train_accuracy)      
      #print("test Loss")
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(epoch+1,
                             train_loss,
                             train_accuracy,
                             loss,
                             accuracy))      
      test_accuracy_arr = []
      test_loss_arr = []
      train_accuracy_arr = [] 
      train_loss_arr = []
      print("1epoch excution time is ",time.time()-time1)                             
print("Execution time") 
print(time.time() - time0)   
      


