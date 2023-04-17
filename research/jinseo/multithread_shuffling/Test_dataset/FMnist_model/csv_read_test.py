import csv
import os
import numpy as np 


f_train = '/home/js/Mnist/integrated_train.csv'
f_test = '/home/js/Mnist/integrated_train.csv'
#df_train = pd.read_csv("/home/js/Mnist/integrated_train.csv")
#df_test = pd.read_csv("/home/js/Mnist/integrated_test.csv")
#train = csv.reader(f_train)
#test = csv.reader(f_test)



with open(f_train,"r") as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    row_count = len(data)

print(row_count)
#print(len(list(test)))