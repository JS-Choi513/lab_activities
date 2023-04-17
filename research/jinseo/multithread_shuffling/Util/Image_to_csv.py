import sys
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
import tensorflow.keras.layers as layers
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import random 
import time 
import os
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import sys
import csv

gpus = tf.config.experimental.list_physical_devices('GPU')


import PIL
import PIL.Image

import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.png'):
  fileList = []
  print(myDir)
  for root, dirs, files in os.walk(myDir, topdown=False):
      for name in files:
          if name.endswith(format):
              fullName = os.path.join(root, name)
              fileList.append(fullName)
  return fileList

# load the original image


myFileList = createFileList('/home/js/horse-or-human/validation/humans/')
myFileList2 = createFileList('/home/js/horse-or-human/validation/horses/')
for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.array(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    
    print(value)
    with open("img_pixels_integ.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

for file in myFileList2:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.array(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    
    print(value)
    with open("img_pixels_integ.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)




df = pd.read_csv('/home/js/noslab.storage/img_pixels_valid.csv')
print(df.shape)      
label1 = np.zeros((255,),dtype = np.int64)
df.insert(0,'label',label1)
#df['label'] = label1
for i in range(128):
      df.iloc[:i,0] = 1

print(df)
print(df.shape)
print(df.head())
print(df.info())
df.to_csv('/home/js/test_hhdata.csv',index=False)
print(df.describe())





'''
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


exit()

train_dir = "/home/js/horse-or-human/train"
test_dir = "/home/js/horse-or-human/validation"

train_humans = os.listdir("/home/js/horse-or-human/train/humans")
train_horses = os.listdir("/home/js/horse-or-human/train/horses")

test_humans = os.listdir("/home/js/horse-or-human/horse-or-human/validation/humans")
test_horses = os.listdir("/home/js/horse-or-human/horse-or-human/validation/horses")


list_ds = tf.data.Dataset.list_files(str(train_humans))
'''






'