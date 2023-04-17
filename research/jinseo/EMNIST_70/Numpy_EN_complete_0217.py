
# -*- coding: utf-8 -*-
import time
import datetime
from numpy import dtype
import tensorflow as tf
#tf.executing_eagerly()
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Model
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
#from tensorflow.python.eager.context import graph_mode
from tensorflow.python.ops.gen_math_ops import mul
import random
from PIL import Image
import matplotlib.pyplot as plt
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#np.set_printoptions(threshold=np.inf, linewidth=np.inf) #release print limit

df_train = pd.read_csv("/home/js/Downloads/emnist-bymerge-train.csv",skiprows=209388)
df_test = pd.read_csv("/home/js/Downloads/emnist-bymerge-test.csv",skiprows=34896)



y_train = df_train.iloc[:,[0]]
x_train = df_train.drop(df_train.iloc[:,[0]],axis=1)
x_train = x_train.drop(x_train.index[0:3])
y_train = y_train.drop(y_train.index[0:3])

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

time0 = time.time() 

y_test = df_test.iloc[:,[0]]
x_test = df_test.drop(df_test.iloc[:,[0]], axis=1)
x_test = x_test.drop(x_test.index[0:2])
y_test = y_test.drop(y_test.index[0:2])

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

np.random.seed(1000)
random.seed(1000)
tf.random.set_seed(1000)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(100, activation='relu')
        self.d2 = Dense(62, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.RMSprop()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#238
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    #images = tf.image.convert_image_dtype(images, tf.float32)
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  test = time.time()
  predictions = model(images)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)
  tf.print("test",time.time()-test)

EPOCHS = 10
for epoch in range(EPOCHS):
  training_time = time.time()
  s = np.arange(488540)
  np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
  train_dx = x_train[s]
  train_dy = y_train[s]    
  batching_img = train_dx.reshape(-1,130,28,28,1)
  batching_lab = train_dy.reshape(-1,130)
  batching_testimg = x_test.reshape(-1,28,28,28,1)
  batching_testlab = y_test.reshape(-1,28)
  for images, labels in zip(batching_img, batching_lab):     
    train_step(images, labels)
  print("1epoch training time", time.time()-training_time)
  
  test_time = time.time()

  for test_images, test_labels in zip(batching_testimg,batching_testlab):
    test_step(test_images, test_labels)

  print("1epoch test time", time.time()-test_time)


  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                      train_loss.result(),
                      train_accuracy.result()*100,
                      test_loss.result(),
                      test_accuracy.result()*100))
print("Execution time") 
print(time.time() - time0)  

