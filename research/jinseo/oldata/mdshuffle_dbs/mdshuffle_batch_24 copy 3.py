
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
THREAD_NUM = 1
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#x_train = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_xtrain.csv')
#y_train = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_ytrain.csv')
#x_test = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_xtest.csv')
#y_test = Data_loader.iter_loadtxt('/home/js/Mnist/integrated_ytest.csv')
df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
#print(x_train.shape)
#print(x_train[0:100])
#print(y_train[0:100])

#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)
lock = threading.Lock()

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


#x_train = np.array(x_train)
#y_train = np.array(y_train)
#x_test = np.array(x_test)
#y_test = np.array(y_test)
#x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train = x_train[..., tf.newaxis]
#x_test = x_test[..., tf.newaxis]
#print(x_train[0:10])
#print(y_train[0:10])

#shuffled_train_result = np.zeros((540000,784,1))
#shuffled_label_result = np.zeros((540000,))
#batching_testimg = x_test.reshape(-1,28,28,1)
#batching_testlab = y_test.reshape(-1,)
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
                  print("loss", loss)
            # print("loss complete")
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)


      @tf.function
      def test_step(self, images, labels):
            predictions = self.call(images)
            t_loss = self.loss_object(labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)
            

model = MyModel()
      
for i in range(THREAD_NUM):
      model_arr.append(MyModel())
      print("add model\n")

for i in range(THREAD_NUM):
      start_idx = offset_idx*train_divide_range
      offset_idx +=1
      end_idx = offset_idx*train_divide_range
      model_idx = i
      # 0: idle , 1: training 
      thread_train_flag.append(0)
      
      train_range_pairs_per_thread.append((start_idx, end_idx, randomidx[i], 128, model_idx))
      print("train_range_pairs_per_thread",train_range_pairs_per_thread[i][0])
      test_range_pairs_per_thread.append(randomidx[i])
      #print("test_range_pairs_per_thread",test_range_pairs_per_thread[i][0])
      


def enqueue_shuffled_data(start,end, randomidx):
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      train, label = part_shuffle(start,end, randomidx)
      print("inint data \n",randomidx)
      print("train shape",train.shape)
      print("label shape",label.shape)
      data =(train, label)
      print("data")
      
      
      
      print("data[0]",data[0][250000])
      #a = pd.DataFrame(data[0])
      #print(a.info)
      
      #print("data[1]",data[1])
      #lock.acquire()
      parted_shuffled_data_q.put(data)
      #lock.relase()
      print("put")
      print("enqueue_shuffled_data Done")

def part_shuffle(start, end, random_idx):      
      print("train_divide_range")
      print("start", start)
      print("end", end)      
      start_point = train_divide_range*random_idx
      print("pointing")
      end_point = start_point+train_divide_range
      print("pointing2")
      tmp_x = x_train[start_point:end_point]
      np.random.shuffle(tmp_x)
      print("positiong")
      tmp_y = y_train[start_point:end_point]
      np.random.shuffle(tmp_y)
      print("positiong2")
      print("tmp_x",tmp_x)
      print("tmp_y",tmp_y)
      #print("shuffle_label_result",shuffled_label_result[0:10])
      print("shuffle1")
      print("shuffle2")
      #print("shuffle_label_result",shuffled_train_result[0:10])
      #print("shuffle_label_result",shuffled_label_result[0:10])
      print("a")
      a = tmp_x.reshape(-1,28,28,1) #shuffled_train_result[start: end]
      #print("reshape")
      print("a",a[0:10])
      print("b")
      b = tmp_y.reshape(-1,)#shuffled_label_result[start: end].reshape(-1,)
      print("b",b[0:10])
      print("reshape2")
     # exit()
      return a, b

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

      for job in parted_shuffled_data_q.queue:
            print("dataq copy while\n")
            #lock.acquire()
            print("THIS IS JOB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #print(job[0][250000])
            data = job
            parted_train_data_q.put(copy.deepcopy(data))
            #lock.release()
      print("parted_train_data_q init is done ")            
      if parted_train_data_q.empty():
            print("parted_train_data_q is empty after mdshuffle")            
                  
      #parted_train_data_q.join()

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
      if  parted_train_data_q.empty():# 바로 훈련모델에 들어가면 되는 데이터 큐 
            print("is it passed?")
            data_q_copy()      
            print("data q copy")
      futures = []
      print("Done for input data queue")
      if parted_train_data_q.empty():
            print("Train Data Q is Empty")            
      with ThreadPoolExecutor(max_workers=threadnum) as executor:            
            for exec_threadnum in range(threadnum):                                                                                                                          
                  futures.append(executor.submit(part_epoch,train_range_pairs_per_thread[exec_threadnum][2],#random index
                                                            train_range_pairs_per_thread[exec_threadnum][3],# batch_size
                                                            train_range_pairs_per_thread[exec_threadnum][4])) #model_index 

      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      #md_shuffle(threadnum) # recharge shuffled Queue                              


def part_epoch(random_idx, batch_size, model_idx):
      print("load Data\n")
      print("init pipeline")
      if parted_train_data_q.empty(): 
            print(" train q is empty")
      else :
            print("not empty",parted_train_data_q.qsize)
      data = parted_train_data_q.get()
      print("Train init data[0]", data[0][250000])        
      print("Train init data[1]", data[1][100:200])        
      print("Train init data[0]", data[0].shape)        

      
      print("Train init data[1]", data[1].shape)  
      train_d = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
      #train_d = tf.data.Dataset.from_tensor_slices(parted_train_data_q.get())      
      train_ds = train_d.batch(32)           
      #print("pipeline train shape",train.shape)
      #print("pipeline label shape",label.shape)
      print("creat pipeline")
      
      for images, labels in train_ds:
            #print(images[100:150])
            #print("model_arr",model_arr[model_idx])
            model.train_step(images, labels)
            
            #print("step train loss",model_arr[model_idx].train_loss.result() )  
      #lock.acquire()            
      train_loss_arr.append(model_arr[model_idx].train_loss.result())          
      print("step train loss",model_arr[model_idx].train_loss.result() )  
      train_accuracy_arr.append(model_arr[model_idx].train_accuracy.result()*100)          
      #lock.release()
      start_point = test_divide_range*random_idx
      end_point = start_point+test_divide_range
      test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg[start_point:end_point],batching_testlab[start_point:end_point])).batch(32)            
      for test_images, test_labels in test_ds:
            model.test_step(test_images, test_labels)      
            #
      lock.acquire()
      test_loss_arr.append(model_arr[model_idx].test_loss.result())#
      test_accuracy_arr.append(model_arr[model_idx].test_accuracy.result()*100)#
      print("step test accuracy",model_arr[model_idx].test_accuracy.result()*100 )
      lock.release()
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! thread epoch Done!!!")
      return True

# 쓰레드별 테스트 시작 전, 해당 쓰레드가 훈련중인지 판별 
def is_thread_training(thread_idx):
      if thread_train_flag[thread_idx] is 0:
            return True 
      else :
            return False             




EPOCHS =1
for epoch in range(EPOCHS):
      multi_epoch(THREAD_NUM)
      avg_test_loss = np.array(test_loss_arr)
      loss = np.mean(avg_test_loss)
      print("test Loss")
      avg_test_accuracy = np.array(test_accuracy_arr)
      accuracy = np.mean(avg_test_accuracy)
      print("test Acc")
      avg_train_loss = np.array(train_loss_arr)
      train_loss = np.mean(avg_train_loss)
      print("train Loss")
      avg_train_accuracy = np.array(train_accuracy_arr)
      train_accuracy = np.mean(avg_train_accuracy)      
      print("test Loss")
      template =  '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      print (template.format(epoch+1,
                             train_loss,
                             train_accuracy,
                             loss,
                             accuracy))      
print("Execution time") 
print(time.time() - time0)   
      


