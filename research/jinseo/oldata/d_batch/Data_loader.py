# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd
# data type, batch size, shuffle, 앙상블, 멀티스레딩
def iter_loadtxt(filename, delimiter=',', skiprows=0,dtype =float) :
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):# index value를 사용하지 않고 range 만큼 반복 iteration을 안함 , skiprows 만큼 건너뀜
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)# '제거 
                for item in line:
                    yield dtype(item)   # 각 split된 원소들이 float자료형으로 변환됨 
        iter_loadtxt.rowlength = len(line)
    data = np.fromiter(iter_func(), dtype=float)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data
    
#time0 = time.time()
#x_train = iter_loadtxt('/home/js/Mnist/integrated_xtrain.csv')
##print(time.time()-time0)
#print("done1\n")
#print(x_train.shape)
#a = pd.DataFrame(x_train)
#b = pd.read_csv('/home/js/Mnist/integrated_xtrain.csv')
#print(a.info())
#print(a.head(5))
#print(x_train.shape)
#print(b.info())
#print(b.head(5))
#time1 = time.time()
#y_train = iter_loadtxt('/home/js/Mnist/integrated_ytrain.csv')
#print(time.time()-time1)
#print("done2\n")
#time2 = time.time()
#x_test = iter_loadtxt('/home/js/Mnist/integrated_xtest.csv')
#print(time.time()-time2)
#print("done3\n")
#time3 = time.time()
#y_test = iter_loadtxt('/home/js/Mnist/integrated_ytest.csv')    
#print(time.time()-time3)
#print("done4\n")
