import numpy as np
import torch
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import torch.nn as nn
from torch import optim
import pdb
import os
import pandas as pd
import random

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay



### 
# Generate Random Number
###

end = [600, 170, 51]
######  1    2    3

# total 75 testing, 746 training
# Category 1: 55, 2: 15, 3: 5

random_list_1 = []
random_list_2 = []
random_list_3 = []
random_list = []

count_1 = 0
count_2 = 0
count_3 = 0

test_list_1 = []
test_list_2 = []
test_list_3 = []
while count_1 < 55:
    r = random.randint(1,600)
    if r not in random_list_1:
        random_list_1.append(r)
        count_1+=1
        total = 0
        for i in range(len(end)):
            total += end[i]
            if r <= total:
                test_list_1.append((i+1, r-total+end[i]))
                break

while count_2 < 15:
    r = random.randint(601,771)
    if r not in random_list_2:
        random_list_2.append(r)
        count_2+=1
        total = 0
        for i in range(len(end)):
            total += end[i]
            if r <= total:
                test_list_2.append((i+1, r-total+end[i]))
                break

while count_3 < 5:
    r = random.randint(772,821)
    if r not in random_list_3:
        random_list_3.append(r)
        count_3+=1
        total = 0
        for i in range(len(end)):
            total += end[i]
            if r <= total:
                test_list_3.append((i+1, r-total+end[i]))
                break            

test_list  = test_list_1 + test_list_2 + test_list_3
random_list = random_list_1 + random_list_2 + random_list_3

random_list.sort()
test_list.sort()

test_input = test_list

print(random_list)
print(test_list)



###
# Load Training and Testing Input
###

end_set = [600, 170, 51]
input_all_set = []
input_train_set = []
input_test_set = []

for i in range (1,4):
    for j in range (1,end_set[i-1]+1):
        if (i,j) not in test_input:
            y = pd.read_csv('/Users/rqzhang/Desktop/Pearl/Input_Raman_Processed_New/'+ str(i) + '_' + str(j) + '.csv', sep=',', header=None)
            y = np.array(y)
            input_train_set.append(np.array(y))
        if (i,j) in test_input: 
            w = pd.read_csv('/Users/rqzhang/Desktop/Pearl/Input_Raman_Processed_New/'+ str(i) + '_' + str(j) + '.csv', sep=',', header=None)
            w = np.array(w)
            input_test_set.append(np.array(w))

input_train_set = np.array(input_train_set)
# input_train_set = np.reshape(input_train_set, (81,250))
input_train_set = np.reshape(input_train_set, (746,250))
print(input_train_set.shape)

input_test_set = np.array(input_test_set)
# input_test_set = np.reshape(input_test_set, (4,250))
input_test_set = np.reshape(input_test_set, (75,250))
print(input_test_set.shape)

print('Finish Loading Training & Testing Input Raman Data')



###
# Select Region of Insterests if neede 
###

## Select First Half of Spectrum
# input_train_set = input_train_set[:,0:125]
# input_test_set = input_test_set[:,0:125]
# print(input_train_set.shape)
# print(input_test_set.shape)
# print('Finish Loading Training & Testing Input Raman Data')

## Select Second Half of Spectrum
# input_train_set = input_train_set[:,124:-1]
# input_test_set = input_test_set[:,124:-1]
# print(input_train_set.shape)
# print(input_test_set.shape)
# print('Finish Loading Training & Testing Input Raman Data')

## Select Highly Weighted Areas
# input_train_set = input_train_set[:, np.r_[0:40, 75:85, 125:135, 155:170, 185:195, 230:245]]
# input_test_set = input_test_set[:, np.r_[0:40, 75:85, 125:135, 155:170, 185:195, 230:245]]
# print(input_train_set.shape)
# print(input_test_set.shape)
# print('Finish Loading Training & Testing Input Raman Data')



###
# Load Training Output (Category)
###
output_train_set = []
output= pd.read_csv('/Users/rqzhang/Desktop/Pearl/Output_Raman_New.csv', sep=',', header=None)
print(output.shape)

output_test_set = []

np_output = np.array(output)
for i in range(len(np_output)):
    if i+1 not in random_list:
        output_train_set.append(np_output[i,:])
    if i+1 in random_list:
        output_test_set.append(np_output[i,:])

output_train_set = np.array(output_train_set)
output_test_set = np.array(output_test_set)
print(output_train_set.shape)
print(output_test_set.shape)
# print(output_test_set)
print('Finish Loading Training Output Category')



###
# SVM Model Prediction
###
X = input_train_set
Y = output_train_set

print(X.shape)
print(Y.shape)
lab = preprocessing.LabelEncoder()
# Y = lab.fit_transform(Y)
# X = lab.fit_transform(X)

# fit the model
# clf = svm.NuSVC(gamma="auto")
clf0 = SVC(kernel="linear", gamma=0.5, C=1.0)
clf0.fit(X, Y)

y_pred = clf0.predict(input_test_set)

print(output_test_set.reshape(75))
print(y_pred)
print('Prediction Accuracy is: ', accuracy_score(output_test_set,y_pred))
