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

df_train = pd.read_csv("/home/js/Mnist/mnist_train1.csv")
df_test = pd.read_csv("/home/js/Mnist/mnist_test1.csv")
#MNIST 원본데이터 실험(1,6)(numpy, tf.data, mdshuffle) shuffle 에 대한 overhead 확인 
#d.batch(mdshuffle+muilt test)
#d.batch(muilt test)만 
#현재 Loss에 대한 issue 있음 

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
os.environ['PYTHONHASHSEED'] = '123'
np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

EPOCHS = 10
THREAD_NUM = 6
model_state = 0

shuffled_train_result = np.zeros((60000,784,1))
shuffled_label_result = np.zeros((60000,))
batching_img = x_train.reshape(-1,28,28,1)
batching_lab = y_train.reshape(-1,)
test_rowcount = len(np.arange(x_test.shape[0]))
test_divide_range = int(test_rowcount/THREAD_NUM)
test_model_arr = []
test_loss_arr = []
test_accuracy_arr = []
# on training : 1, idle : 0

os.environ['PYTHONHASHSEED'] = '123'
np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

# d.batch -> i.batch 
#비교군 numpy, tf.data, i.data (dbatch, mdshuffle 삭제)
#tf.data, idata는 쓰레드 6개도 추가 
# thead 1개일 때, 6개일 때(numpy는 제외) 
#mnist_original
#fashion mnist_original
#mnist 1GB
#fashion mnist_original 

def multi_shuffling(train_arr,label_arr,threadnum):
      rowcount = len(np.arange(train_arr.shape[0]))  
      global divide_range
      divide_range = int(rowcount/threadnum)
      global train
      train = train_arr
      global label
      label = label_arr
      randomidx = random.sample(range(threadnum),threadnum)
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            offset_idx = 0
            for exec_threadnum in range(threadnum):
                  start_idx = offset_idx*divide_range
                  offset_idx +=1
                  end_idx = offset_idx*divide_range
                  executor.submit(part_shuffle,start_idx, end_idx, randomidx[exec_threadnum])                
      img = shuffled_train_result.reshape(-1,28,28,1)
      lab = shuffled_label_result.reshape(-1,)
      return img, lab


def part_shuffle(start, end, random_idx):
      

      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      start_point = divide_range*random_idx
      end_point = start_point+divide_range
      tmp_x = train[start_point:end_point]
      tmp_y = label[start_point:end_point]      
      shuffled_train_result[start: end] = tmp_x[suffleidx]
      shuffled_label_result[start: end] = tmp_y[suffleidx]

batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)


'''tf.data'''
'''tf.data'''


class MyModel(Model): #keras의 모델을 상속받음 
      def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10, activation='softmax')
            
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam()
            #self.optimizer = tf.keras.optimizers.SGD()
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            #self.train_loss = []
            #self.test_loss = []
            #self.train_accuracy = []
            #self.test_accuracy = []
      
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

evt = threading.Event()


class StateError(Exception):
      def __init__(self, msg='Model on training...'):
            self.msg = msg
      def __str__(self):
            return self.msg

def model_train(threadnum, batch_size):
#      evt.clear()

      time1 = time.time()
     # print("model train called")

      #train_d = tf.data.Dataset.from_tensor_slices(multi_shuffling(x_train,y_train,threadnum))
      train_d = tf.data.Dataset.from_tensor_slices((batching_img,batching_lab))
      train_ds = train_d.batch(batch_size)
      for images, labels in train_ds:
            model.train_step(images, labels)
            #print("train loss", model.train_loss.result())
      evt.set()            
      #print("after train model_state:", model_state) 
      print("model_training time: ", time.time()-time1)

def test_model_copy(threadnum, model):

      for i in range(threadnum):
           # print("copy model")
            test_model_arr.append(copy.copy(model))
      global model_state
      model_state = 0

def multi_test(threadnum, batch_size):
      evt.wait()
      #print("thread 종료됨 ")
      #print("multitest state", model_state)
      #if model_state < 2:
      #      raise StateError()
      #else:
      test_model_copy(6, model)        # 여기서 0으로 변함   
      futures = []
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            for exec_threadnum in range(threadnum):
                  futures.append(executor.submit(part_test,exec_threadnum, batch_size))                         
               #   print("test thread deploy")
            #print("thread thread called")                        
            thread_restart(threadnum, 128)                 
      wait(futures,timeout=None,return_when=ALL_COMPLETED)
      evt.clear()

def part_test(model_idx, batch_size):
      
      start = test_divide_range*model_idx
      end = start+test_divide_range              
      test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg[start: end], batching_testlab[start: end])).batch(batch_size)
      for test_images, test_labels in test_ds:
            test_model_arr[model_idx].test_step(test_images, test_labels)
      test_loss_arr.append(test_model_arr[model_idx].test_loss.result())
      test_accuracy_arr.append(test_model_arr[model_idx].test_accuracy.result()*100)       


#훈련 쓰레드 
train_thread = threading.Thread(target=model_train, args=(THREAD_NUM, 128))

#Epoch 종료 후 훈련쓰레드 초기화, 재시작 
def thread_restart(threadnum, batch_size):
      global train_thread
      train_thread.join()
      train_thread = threading.Thread(target=model_train, args=(threadnum, batch_size))
      train_thread.start()

for epoch in range(EPOCHS):
      time1 = time.time()
      if epoch == 0:# 초기상태일 경우 훈련thread 먼저 시작 
            train_thread.start()
      multi_test(THREAD_NUM, 32)
      #global loss, accuracy 
      template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
      avg_test_loss = np.array(test_loss_arr)
      test_loss = np.mean(avg_test_loss)
      avg_test_accuracy = np.array(test_accuracy_arr)
      test_accuracy = np.mean(avg_test_accuracy)

      print (template.format(epoch+1,
                              model.train_loss.result(),
                              model.train_accuracy.result()*100,
                              test_loss,
                              test_accuracy))                         
      print("1epoch : ", time.time()-time1)

print("Execution time") 
print(time.time() - time0)  

