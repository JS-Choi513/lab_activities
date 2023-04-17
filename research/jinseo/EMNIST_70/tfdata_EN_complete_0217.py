
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import glob 
import random
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


time0 = time.time() 

batching_img = x_train.reshape(-1,28,28,1)
batching_lab = y_train.reshape(-1,)
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)


# tf.data data pipeline 
traind = tf.data.Dataset.from_tensor_slices((batching_img, batching_lab)).shuffle(348968)
train_ds = traind.batch(130)
test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(28)

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
        train_step(images, labels)   
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)  


          
    template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
    print (template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))


print("Elapsed time: ",time.time() - time0)  


