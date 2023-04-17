import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
from concurrent.futures import ThreadPoolExecutor
import sys

'''
filename = '/media/js/test/cpu.txt'
def logWrite(data):    
  f = open(filename,"a")
  f.write(data)
  f.close()
time0 = time.time() 
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
print(x_train.dtype)
random.seed(1000)

shuffled_train_result = np.zeros((540000,784,1))
shuffled_label_result = np.zeros((540000,))

'''

arr2 = np.full((540000,784,1),255.485,dtype=np.float64)

arr = np.array(range(10))
arr = arr.reshape(1,10)
arrt = tf.convert_to_tensor(arr)
arr_result = np.arange(10)#tf.convert_to_tensor(arr)
shuffled_train_result = np.zeros((540000,784,1))
gpus = tf.config.experimental.list_physical_devices('GPU')
            
def rrr(x):
      print(x)
      #sys.stdout.write(x)
      y=x+10
      return y

def multi_shuffling(train_arr ,threadnum=10):
      print("type",type(train_arr))
      print("convert to tensor",train_arr)
      #rowcount = int(tf.shape(train_arr))
      rowcount = len(np.arange(train_arr.shape[0]))   
      print("convert to tensor22222222222",rowcount)
      
      global tf_to_numpy 
      tf_to_numpy = train_arr#.numpy()
      global divide_range 
      divide_range = int(rowcount/threadnum)
      print("divide_range",divide_range)
      randomidx = random.sample(range(threadnum),threadnum)

      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            offset_idx = 0
            for exec_threadnum in range(threadnum):
                  sys.stdout.write("ddddd\n")
                  start_idx = offset_idx*divide_range
                  offset_idx +=1
                  end_idx = offset_idx*divide_range
                  executor.submit(part_shuffle,start_idx, end_idx, randomidx[exec_threadnum])                
      return tf.convert_to_tensor(arr_result)

def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      start_point = divide_range*random_idx
      end_point = start_point+divide_range
      tmp_x = tf_to_numpy[start_point:end_point]           
      arr_result[start: end] = tmp_x[suffleidx]
      #sys.stdout.write("ddddd")
      #sys.stdout.write(arr_result[start: end])
      

#############################################################################
#tf.data test 
'''
데이터셋과 동일한 크기의 numpy 배열 54000,784,1을 생성해서 
각각의 shuffle의 수행시간을 측정하였다. 
그 결과 
numpy < multi shuffling < tf.data 순으로 셔플이 빨랐다. 
0.3          0.9            1.1

pandas를 사용해서 스토리지에서 데이터를 로드하는 과정을 추가했을 경우
multi shuffling < numpy < tf.data 순으로 셔플이 빨랐다. 
        10          12       21
        dho?
        
'''
#############################################################################

np.set_printoptions(precision=4)
'''
텐서플로우 객체끼리 assign은 지원하지 않음 
arrt1 = tf.convert_to_tensor(arr)
print("aa")
print(arrt[1:3])
arrt[1:3] = arrt1[5:7]
print(arrt[1:3])

'''
#multi_shuffling(arrt,1)

dataset = tf.data.Dataset.from_tensor_slices(arr)
shuffled = dataset.map(lambda x: tf.numpy_function(func = multi_shuffling, inp = [x],Tout=tf.int64))#.shuffle(10)
#shuffled = dataset.map(lambda x: tf.py_function(func = rrr, inp = [x],Tout=tf.int64))#.shuffle(10)
print(dataset)

print(list(shuffled.as_numpy_iterator()))


for i in range(5):
      for elem in shuffled:
            print(elem.numpy())
      print("Epoch")
      #multi_shuffling(arr,10)
      #dataset = tf.data.Dataset.from_tensor_slices(arr_result)#.shuffle(10)
      #dataset = tf.data.Dataset.from_tensor_slices(arr_result).
"""


arr = np.arange(100000)
time1 = time.time()
print("init time", time.time()-time1)
#print(arr)



#''' case 1. tf.data.shuffle '''
#time2 = time.time()
#dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(540000).batch(32)
#print("shuffle time", time.time()-time2)
#1.18
#10.48
#23


#time3 = time.time()
#for elem in dataset:
#      print(elem.numpy())
#print("shuffle time", time.time()-time2)

#''' case 2. numpy shuffle'''
time4 = time.time()
np.random.seed(1000)
s = np.arange(arr2.shape[0])
np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
train_dx = arr2[s]
print("numpy shuffle time", time.time()-time4)
#0.4
#10.9
#12

#''' case 3. multi-thread shuffle'''
#time5 = time.time()
#multi_shuffling(arr2,1000)
#print("multi-thread shuffle time", time.time()-time5)
#0.9
#9.4
#10
'''





#arr2 = arr2.reshape(540000,784)
#df = pd.DataFrame(arr2)
#df.to_csv('test.csv',index=True)
"""


