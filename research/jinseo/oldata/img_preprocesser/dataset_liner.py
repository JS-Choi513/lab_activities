import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imagesize
import os
import shutil
import pathlib
import PIL

root= '/home/js/Downloads/CUB_200_2011/images'
specific = root+'/001.Black_footed_Albatross'

data_dir = pathlib.Path(root)
image_count = list(data_dir.glob('*/*.jpg'))
print(image_count)# whole image number
exit()
albatross = list(data_dir.glob('*/*.jpg'))
print(PIL.Image.open(str(albatross[0])))
#plt.imshow(PIL.Image.open(str(albatross[0])))
#plt.show()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2,
                                                               subset="training",seed=123,
                                                               image_size=(255,255),batch_size=32)
                                                               
class_names = train_ds.class_names
print(class_names)


for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

exit()
print(len(os.listdir(specific)))
print(os.listdir(specific)[:10])
#tf.keras.preprocessing.image.load_img()
tf.keras.preprocessing.image_dataset_from_directory()

img = tf.keras.preprocessing.image.load_img(specific+'/Black_Footed_Albatross_0001_796111.jpg',target_size = (250,250))
img_tensor = tf.keras.preprocessing.image.img_to_array(img)
print(img_tensor)

