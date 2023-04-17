
# -*- coding: utf-8 -*-
import time
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
#from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.python.eager.context import graph_mode
from tensorflow.python.ops.gen_math_ops import mul
device_lib.list_local_devices()
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import glob 
import random
import threadpool
import copy
from tensorflow.python.framework.ops import disable_eager_execution
import cProfile
import re
#disable_eager_execution()

filename = '/media/js/test/cpu.txt'
def logWrite(data):    
  f = open(filename,"a")
  f.write(data)
  f.close()
time0 = time.time() 
df_train = pd.read_csv("/home/js/Mnist/mnist_train1.csv")
df_test = pd.read_csv("/home/js/Mnist/mnist_test1.csv")
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
random.seed(50)

shuffled_train_result = np.zeros((660000,784,1))
shuffled_label_result = np.zeros((660000,))
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)



s = np.arange(x_train.shape[0])
np.random.shuffle(s)# 6만라인 범위 랜덤셔플링 0.35
train_dx = x_train[s]
train_dy = y_train[s]




batching_img = x_train.reshape(-1,28,28,1)
batching_lab = y_train.reshape(-1,)
batching_testimg = x_test.reshape(-1,28,28,1)
batching_testlab = y_test.reshape(-1,)

'''tf.data'''
time1 = time.time()         
#pr = cProfile.Profile()
#pr.enable()
train_ds = tf.data.Dataset.from_tensor_slices((batching_img, batching_lab)).shuffle(60000).batch(32)#.shuffle(10000)
test_ds = tf.data.Dataset.from_tensor_slices((batching_testimg,batching_testlab)).batch(32)
print("shuffling time",time.time() - time1)  

'''tf.data'''

#cProfile.run('re.compile("foo|bar")')

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
'''
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
tf.summary.trace_on(graph=False, profiler=True)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
'''
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
  #for images, labels in zip(batching_img, batching_lab):
  #  train_step(images, labels)
#    '''tf.data 사용 시 주석해제 '''
  for images, labels in train_ds:
        train_step(images, labels)
  '''
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
  '''

    
    

#  for test_images, test_labels in zip(batching_testimg,batching_testlab):
#    test_step(test_images, test_labels)
#   '''tf.data 사용 시 주석해제 '''
  for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
  '''
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
  '''  

    
    
  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()
#pr.disable()

#pr.dump_stats('logs.txt')                         
#with train_summary_writer.as_default(): #텐서 보드 (그래프 추가)
#  tf.summary.trace_export(
#      name='graph',
#      step=0,
#      profiler_outdir='logs/gradient_tape/' + current_time + '/train')
print("Execution time") 
print(time.time() - time0)  

