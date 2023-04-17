# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ConvLSTM2D
from tensorflow.keras import Model
import os
import time
import numpy as np
from tensorflow.python.client import device_lib
#from tensorflow.python.eager.context import graph_mode
from tensorflow.python.ops.gen_math_ops import mul
import random
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

# set the train dataset path
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
time0 = time.time() 

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
random.seed(1000)

shuffled_train_result = np.zeros((60000,784,1))
shuffled_label_result = np.zeros((60000,))

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

#@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

#@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 10
for epoch in range(EPOCHS):
  s = np.arange(x_train.shape[0])
  np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
  train_dx = x_train[s]
  train_dy = y_train[s]    
  batching_img = x_train.reshape(-1,125,28,28,1)
  batching_lab = y_train.reshape(-1,125)
  batching_testimg = x_test.reshape(-1,25,28,28,1)
  batching_testlab = y_test.reshape(-1,25)
  for images, labels in zip(batching_img, batching_lab):        
    train_step(images, labels)

    
  for test_images, test_labels in zip(batching_testimg,batching_testlab):
    test_step(test_images, test_labels)
      
  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                      train_loss.result(),
                      train_accuracy.result()*100,
                      test_loss.result(),
                      test_accuracy.result()*100))
print("Execution time") 
print(time.time() - time0)  

