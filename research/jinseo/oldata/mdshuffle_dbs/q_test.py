import time
from tensorflow.python.framework.func_graph import ALLOWLIST_COLLECTIONS
from tensorflow.python.keras.backend import _broadcast_normalize_batch_in_training
from tensorflow.python.ops.gen_dataset_ops import interleave_dataset
import os
import time
import math
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.python.ops.gen_math_ops import mul
from tensorflow.python.ops.gen_resource_variable_ops import mutex_lock
device_lib.list_local_devices()
import threading
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
import random
import threading 
from queue import Queue
from numpy import genfromtxt
import tensorflow as tf
import MNIST_model as MNIST
import Data_loader
import copy

q1 = Queue()


lock = threading.Lock()

#print(q1.get())
#print(q1.get())
#print(q1.get())
#print(q1.get())
#print(q1.get())


arr = np.array(range(50000),dtype=np.float16)
rnd_idx = [5,2,1,3,4,0]
offset_idx=0
train_divide_range= int(50000/6)
test_range_pairs_per_thread = []
for i in range(6):
      start_idx = offset_idx*train_divide_range
      offset_idx +=1
      end_idx = offset_idx*train_divide_range
      model_idx = i
      test_range_pairs_per_thread.append((start_idx,end_idx))


def enqueue_shuffled_data(start,end, randomidx):
      print("inint data \n",randomidx)
      #print("train shape",train.shape)
      #print("label shape",label.shape)
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
      a = arr[start:end]
      b = rnd_idx[randomidx]
      data = a
      print(data)
      lock.acquire()
      q1.put_nowait(data)
      lock.release()


futures = []
with ThreadPoolExecutor(max_workers=6) as executor:
    idx=0
    for exec_threadnum in range(6):
            futures.append(executor.submit(enqueue_shuffled_data,test_range_pairs_per_thread[exec_threadnum][0],test_range_pairs_per_thread[exec_threadnum][1],0)) 
            print(test_range_pairs_per_thread[exec_threadnum][0]) 
            print(test_range_pairs_per_thread[exec_threadnum][1]) 
            print("")
wait(futures,timeout=None,return_when=ALL_COMPLETED)

for i in q1.queue:
    data=i
    print(data)
    print("")

if q1.empty():
    print("empty")   