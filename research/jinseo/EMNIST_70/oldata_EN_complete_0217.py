
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Model
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
import threading
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
import random
import copy

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Train data load from CSV 
df_train = pd.read_csv("/home/js/Downloads/emnist-bymerge-train.csv",skiprows=209388)
df_test = pd.read_csv("/home/js/Downloads/emnist-bymerge-test.csv",skiprows=34896)
print(df_train.shape)
print(df_test.shape)

y_train = df_train.iloc[:,[0]]
x_train = df_train.drop(df_train.iloc[:,[0]],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_test = df_test.iloc[:,[0]]
x_test = df_test.drop(df_test.iloc[:,[0]], axis=1)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
random.seed(1000)
avg_train_t = []
avg_test_t = []

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
THREAD_NUM = 24
TRAIN_BATCH_SIZE = 130
TEST_BATCH_SIZE = 28
time0 = time.time() 

batching_img = x_train.reshape(-1,28,28,1)
batching_lab = y_train.reshape(-1,)
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)
test_rowcount = len(np.arange(x_test.shape[0]))
test_divide_range = int(test_rowcount/THREAD_NUM)
test_model_arr = []
test_loss_arr = []
test_accuracy_arr = []

# tf.data data pipeline 

traind = tf.data.Dataset.from_tensor_slices((batching_img, batching_lab)).shuffle(348968)
train_ds = traind.batch(130)

class MyModel(Model):
      def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(16, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(100, activation='relu')
            self.d2 = Dense(62, activation='softmax')  

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

@tf.function
def train_step(images, labels):
      with tf.GradientTape() as tape:
            predictions = model(images)
            loss = model.loss_object(labels, predictions)        
      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      model.train_loss(loss)
      model.train_accuracy(labels, predictions)

@tf.function
def test_step(Mymodel,images, labels):
      predictions = Mymodel(images)
      t_loss = Mymodel.loss_object(labels, predictions)
      Mymodel.test_loss(t_loss)
      Mymodel.test_accuracy(labels, predictions)

# CNN model global object            
model = MyModel()

# Train thread, Test thread event call object
evt = threading.Event()

EPOCHS = 10

def model_train(epoch):
      #evt.set()            
      for images, labels in train_ds:
            train_step(images, labels)            
      
      


#Epoch 종료 후 훈련쓰레드 초기화, 재시작 
def thread_restart(epoch):
      global train_thread
      #train_thread.join()
      train_thread = threading.Thread(target=model_train, args=(epoch,))
      train_thread.start()


def part_test(model_idx, batch_size):
      start = test_divide_range*model_idx
      end = start+test_divide_range    
      test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg[start: end], batching_testlab[start: end])).batch(batch_size)
      
      for test_images, test_labels in test_ds:
            test_step(copied_model, test_images, test_labels)
      
      test_loss_arr.append(copied_model.test_loss.result())
      test_accuracy_arr.append(copied_model.test_accuracy.result()*100)      

futures = []

with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:  
      for epoch in range(EPOCHS):
            if epoch == 0:
                  thread_restart(epoch)         
            train_thread.join()
            copied_model = copy.copy(model)
            for exec_threadnum in range(THREAD_NUM):
                  futures.append(executor.submit(part_test,exec_threadnum, 28))
            if epoch < EPOCHS and epoch > 0:                                 
                  thread_restart(epoch)          
            wait(futures,timeout=None,return_when=ALL_COMPLETED)
            futures.clear()
 
            
            template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'

            if THREAD_NUM==1:
                        print (template.format(epoch+1,
                                    model.train_loss.result(),
                                    model.train_accuracy.result()*100,
                                    copied_model.test_loss.result(),
                                    copied_model.test_accuracy.result()*100)   )
            else:
                  avg_test_loss = np.array(test_loss_arr) 
                  test_loss = np.mean(avg_test_loss)
                  avg_test_accuracy = np.array(test_accuracy_arr)
                  test_accuracy = np.mean(avg_test_accuracy)

                  print (template.format(epoch+1,
                                          model.train_loss.result(),
                                          model.train_accuracy.result()*100,
                                          test_loss,
                                          test_accuracy))   
            test_loss_arr.clear()
            test_accuracy_arr.clear()
      print("Elapsed time: ",time.time() - time0)  


