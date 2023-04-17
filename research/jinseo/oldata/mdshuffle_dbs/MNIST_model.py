# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model): #keras의 모델을 상속받음 
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

      # Model 클래스의 전역변수로 쓰는게 옳은 방법인가?      
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
      optimizer = tf.keras.optimizers.Adam()
      train_loss = tf.keras.metrics.Mean(name='train_loss')
      train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
      test_loss = tf.keras.metrics.Mean(name='test_loss')
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


      @tf.function
      def train_step( images, labels, model_arr, model_idx):
            with tf.GradientTape() as tape:
                  predictions = model_arr[model_idx](images) # 쓰레드마다 할당된 모델 객체에 대해 예측 
                  loss = MyModel.loss_object(labels, predictions)
            gradients = tape.gradient(loss, model_arr[model_idx].trainable_variables)
            MyModel.optimizer.apply_gradients(zip(gradients, model_arr[model_idx].trainable_variables))
            MyModel.train_loss(loss)
            MyModel.train_accuracy(labels, predictions)


      @tf.function
      def test_step(images, labels, model_arr, model_idx):
            predictions = model_arr[model_idx](images)
            t_loss = MyModel.loss_object(labels, predictions)

            MyModel.test_loss(t_loss)
            MyModel.test_accuracy(labels, predictions)



