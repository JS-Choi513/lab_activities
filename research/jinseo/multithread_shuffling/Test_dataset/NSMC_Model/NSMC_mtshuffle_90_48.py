import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import random 
import time 
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
shuffled_train_result = np.zeros((145393,30))
shuffled_label_result = np.zeros((145393,1))

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
SEED = 90
set_seeds(SEED)
time1 = time.time()
def multi_shuffling(train_arr,label_arr,threadnum):
      time1 = time.time()
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
      img = shuffled_train_result#.reshape(-1,28,28,1)
      lab = shuffled_label_result#.reshape(-1,)
      print("shuffling time",time.time()-time1)
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

x_train = pd.read_csv("/home/js/movie_review_trainx.csv")
y_train = pd.read_csv("/home/js/movie_review_trainy.csv")
x_test = pd.read_csv("/home/js/movie_review_testx.csv")
y_test = pd.read_csv("/home/js/movie_review_testy.csv")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()


test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(43752,100)
        self.lstm = LSTM(128)
        self.dense = Dense(8, activation='relu')    
        self.dense2 = Dense(1,activation='sigmoid')
        
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dense(x)
        return self.dense2(x)

model = MyModel()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

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
  train_ds = tf.data.Dataset.from_tensor_slices(multi_shuffling(x_train,y_train,48)).batch(32)
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
  
print("Elapsed Time",time.time()-time1)

# 100 1450

