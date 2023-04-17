
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
import random
import threading 
from queue import Queue
from numpy import genfromtxt
import tensorflow as tf
import MNIST_model as MNIST
import Data_loader
import copy
THREAD_NUM = 6



df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
THREAD_NUM = 6




time0 = time.time() 
y_train = df_train['label']
x_train = df_train.drop(['label'],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = df_test['label']
x_test = df_test.drop(['label'], axis=1)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
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
train_range_pairs_per_thread = []
test_range_pairs_per_thread =[]
train_rowcount = len(np.arange(x_train.shape[0]))  
test_rowcount = len(np.arange(x_test.shape[0]))
randomidx = random.sample(range(THREAD_NUM),THREAD_NUM)
thread_train_flag = []
offset_idx = 0
lock = threading.Lock()

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
print("train_divide_range",train_divide_range)

for i in range(THREAD_NUM):
      model = MNIST.MyModel()
      model_arr.append(MNIST.MyModel())
      print("add model\n")

for i in range(THREAD_NUM):
      start_idx = offset_idx*train_divide_range
      offset_idx +=1
      end_idx = offset_idx*train_divide_range
      model_idx = i
      # 0: idle , 1: training 
      thread_train_flag.append(0)
      if i < 5:
            train_range_pairs_per_thread.append((start_idx, end_idx, randomidx[i], 256, model_idx))
            print(train_range_pairs_per_thread[i][0])
      else: 
            train_range_pairs_per_thread.append((start_idx, end_idx, randomidx[i], 256, model_idx))
      test_range_pairs_per_thread.append(randomidx[i])


def enqueue_shuffled_data(start,end, randomidx):
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      train, label = part_shuffle(start,end, randomidx)
      print("partshuffle end")
      data = (train, label)
      print(data)
      lock.acquire()
      parted_shuffled_data_q.put_nowait(data)
      lock.release()


def get_part_data(start, end):
      train = x_train[start:end].reshape(-1,28,28,1)
      lab = y_train[start:end].reshape(-1,)
      return train, lab

def data_q_copy():
      #초기상태 
      print("data q init")
      if parted_train_data_q.empty() and parted_shuffled_data_q.empty():
            print("init stat init\n")
            md_shuffle(THREAD_NUM)
      if parted_train_data_q.empty():
            print("parted_train_data_q is empty after mdshuffle")            
      
      for job in parted_shuffled_data_q.queue:
            print("dataq copy while\n")
            lock.acquire()
            parted_train_data_q.put(job)
            lock.release()
      #parted_train_data_q.join()

def set_shuffled_data(start, end, data):
      shuffled_train_result[start: end] = data
      shuffled_label_result[start: end] = data



def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(train_divide_range) # 0 ~ 90000
      print("train_divide_range")
      print("start", start)
      print("end", end)
      np.random.shuffle(suffleidx)    
      start_point = train_divide_range*random_idx
      print("pointing")
      end_point = start_point+train_divide_range
      print("pointing2")
      tmp_x = x_train[start_point:end_point]
      print("positiong")
      tmp_y = y_train[start_point:end_point]      
      print("positiong2")
      shuffled_train_result[start: end] = np.random.shuffle(tmp_x)
      print("shuffle1")
      shuffled_label_result[start: end] = np.random.shuffle(tmp_y)
      print("shuffle2")
      a = shuffled_train_result[start: end].reshape(-1,28,28,1)
      print("reshape")
      
      b = shuffled_label_result[start: end].reshape(-1,)
      print("reshape2")
      return a, b

def md_shuffle(threadnum):
      futures = []
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(enqueue_shuffled_data, train_range_pairs_per_thread[exec_threadnum][0],     # train_data start pointer
                                                             train_range_pairs_per_thread[exec_threadnum][1],     # train_data end pointer
                                                             train_range_pairs_per_thread[exec_threadnum][2]))     # randon index for merge shuffled data   
                  print(train_range_pairs_per_thread[exec_threadnum][0])                                                             
                  print(train_range_pairs_per_thread[exec_threadnum][1])       
                  print(train_range_pairs_per_thread[exec_threadnum][2])       
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      return True


def multi_epoch(threadnum):
      print("multi epoch init")
      if  parted_train_data_q.empty():
            print("is it passed?")
            data_q_copy()      
            print("data q copy")
      print("Done for input data queue")
      if parted_train_data_q.empty():
            print("zbdbnsdfsdgdhfsdfgdfgsdfgdfgsdfgdfgsfdg")            
      for data in parted_train_data_q.queue:
            print("parted_train_data_q.qeue")
            print(data[0].shape)
            print(data[1].shape)


EPOCHS = 10
for epoch in range(EPOCHS):
      multi_epoch(THREAD_NUM)
      avg_test_loss =   np.array(test_loss_arr)
      loss = np.mean(avg_test_loss)
      avg_test_accuracy =   np.array(test_accuracy_arr)
      accuracy = np.mean(avg_test_accuracy)
      avg_train_loss = np.array(train_loss_arr)
      train_loss = np.mean(avg_train_loss)
      avg_train_accuracy = np.array(train_accuracy_arr)
      train_accuracy = np.mean(avg_train_accuracy)      
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(epoch+1,
                             train_loss,
                             train_accuracy,
                             loss,
                             accuracy))      
print("Execution time") 
print(time.time() - time0)   
      


