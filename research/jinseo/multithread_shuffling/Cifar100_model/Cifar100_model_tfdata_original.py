
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D,Dropout
from tensorflow.keras import Model
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.keras.engine.input_spec import assert_input_compatibility
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import glob 
import random
import threadpool
import copy
import matplotlib.pyplot as plt

(x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
time0 = time.time()
'''
x_train1 = np.append(x_train1,x_train1, axis=0)
x_train1 = np.append(x_train1,x_train1, axis=0)
y_train1 = np.append(y_train1,y_train1, axis=0)
y_train1 = np.append(y_train1,y_train1, axis=0)
x_test1 = np.append(x_test1,x_test1,axis=0)
x_test1 = np.append(x_test1,x_test1,axis=0)
y_test1 = np.append(y_test1,y_test1,axis=0)
y_test1 = np.append(y_test1,y_test1,axis=0)


'''
print(x_train1.shape)
print(y_train1.shape)
print(x_test1.shape)
print(y_test1.shape)
x_train1 = x_train1/255
x_test1 = x_test1/255


'''
filename = '/media/js/test/cpu.txt'
def logWrite(data):    
  f = open(filename,"a")
  f.write(data)
  f.close()
time0 = time.time() 
df_train = pd.read_csv("/home/js/train.csv",index_col=False)
df_test = pd.read_csv("/home/js/test.csv",index_col=False)
df_train_label = pd.read_csv("/home/js/train_label.csv",index_col=False)
df_test_label = pd.read_csv("/home/js/test_label.csv",index_col=False)
y_train = df_train_label
#y_train = df_train['label']
#y_train = y_train.iloc[:50000]
x_train = df_train.drop(['label'],axis=1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
#y_test = df_test['label']
y_test = df_test_label
#y_test = y_test.iloc[:10000]
x_test = df_test.drop(['label'], axis=1)
print("xtest",x_test)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

'''

random.seed(1000)


gpus = tf.config.experimental.list_physical_devices('GPU')
         
'''tf.data'''
# multishuffle + tf.data
#train_ds = tf.data.Dataset.from_tensor_slices((batch_train_x, batch_train_y)).batch(32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train1, y_train1)).shuffle(200000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test1, y_test1)).batch(32)
'''tf.data'''


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=(3,3), activation='relu' , input_shape=(32,32,3))
        self.flatten = Flatten()        
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(100, activation='softmax')
                
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
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
  for images, labels in train_ds:
        #print("train shape", images.shape, labels.shape)
        train_step(images, labels)

  for test_images, test_labels in test_ds:
        #print("test shape", test_images.shape, test_labels.shape)
        test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))                         
  

print("Execution time") 
print(time.time() - time0)  

