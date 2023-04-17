import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
from concurrent.futures import ThreadPoolExecutor
import sys
import copy

random.seed(500)
raw_arr = np.array(range(50))


shuffled_train_result = np.arange(50)
shuffled_train_result = np.zeros((50))

global rere 
rere = np.array(range(50))

def multi_shuffling(train_arr,threadnum=10):
      rowcount = len(np.arange(train_arr.shape[0]))   
      global divide_range
      global temp_arr 
      temp_arr = train_arr
      divide_range = int(rowcount/threadnum)
      randomidx = random.sample(range(threadnum),threadnum)
      with ThreadPoolExecutor(max_workers=threadnum) as executor:
            offset_idx = 0
            for exec_threadnum in range(threadnum):
                  #sys.stdout.write("ddddd\n")
                  start_idx = offset_idx*divide_range
                  offset_idx +=1
                  end_idx = offset_idx*divide_range
                  executor.submit(part_shuffle,start_idx, end_idx, randomidx[exec_threadnum])     
      rere = copy.deepcopy(shuffled_train_result)
      return rere
      
def part_shuffle(start, end, random_idx):
      suffleidx = np.arange(divide_range)
      np.random.shuffle(suffleidx)    
      start_point = divide_range*random_idx
      end_point = start_point+divide_range
      tmp_x = temp_arr[start_point:end_point]
      shuffled_train_result[start: end] = tmp_x[suffleidx]




shufflearr = multi_shuffling(raw_arr,5)
dataset = tf.data.Dataset.from_tensor_slices(shufflearr).batch(10) # batch_size가 32인 tf.data 객체 생성
print(dataset) # 객체정보 출력 

for epoch in range(5): # Epoch = 5
      for batch in dataset: # batch call 
            print("BATCH_result",batch) #
      shufflearr = copy.deepcopy(multi_shuffling(raw_arr,5))
      print("")
      print("Epoch")            







