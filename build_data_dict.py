# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:27:27 2017

@author: HGY
"""

import csv
import numpy as np
import numpy.matlib
import pickle
from scipy.io import savemat
from sklearn import mixture
import os

FEA_ROOT = '../features/ComPare_2016/LLD/'
TRAIN_TARGET = 'train_final_large'
TEST_TARGET = 'test10'
'''
### ----------------------------  Train Data  --------------------------------
TRAIN_PATH = ('../wav/'+TRAIN_TARGET+'/')
TrainList = []
for root, dirs, files in os.walk(TRAIN_PATH):
    for file in files:
        if file.endswith(".wav"):
             TrainList.append(os.path.split(file)[1])

# build lld feature dict
DATA_TRAIN_PATH = FEA_ROOT+TRAIN_TARGET+'.csv'
Data_Train = {k.split('.')[0]:[] for k in TrainList}

f = open(DATA_TRAIN_PATH, 'r')  
reader = csv.reader(f)
headers = reader.next()
FeaName = ';'.join(headers).split(';')
print 'The dimension of the feature is',len(FeaName)-1

for row in reader:  
    r = ';'.join(row).split(';')
    key = r[0].replace('\'','').split('.')[0]
    val = np.array(r[1:]).astype('float')
    Data_Train[key].append(val)
    print 'train',key
f.close()
Data_Train = dict(map(lambda (k,v): (k, np.array(v)), Data_Train.iteritems()))

# Output as pickle
with open(FEA_ROOT+'lld_'+TRAIN_TARGET+'.pickle', 'wb') as handle:
    pickle.dump(Data_Train, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''


### ----------------------------  Test Data  --------------------------------
TEST_PATH = ('../wav/test/'+TEST_TARGET+'/')
TestList = []
for root, dirs, files in os.walk(TEST_PATH):
    for file in files:
        if file.endswith(".wav"):
             TestList.append(os.path.split(file)[1])


# build lld feature dict
DATA_TEST_PATH = FEA_ROOT+'/test/'+TEST_TARGET+'.csv'
Data_Test = {k.split('.')[0]:[] for k in TestList}

f = open(DATA_TEST_PATH, 'r')  
reader = csv.reader(f)
headers = reader.next()
FeaName = ';'.join(headers).split(';')
print 'The dimension of the feature is',len(FeaName)-1

for row in reader:  
    r = ';'.join(row).split(';')
    key = r[0].replace('\'','').split('.')[0]
    val = np.array(r[1:]).astype('float')
    Data_Test[key].append(val)
    print 'test',key
f.close()
Data_Test = dict(map(lambda (k,v): (k, np.array(v)), Data_Test.iteritems()))

# Output as pickle
with open(FEA_ROOT+'/test/lld_'+TEST_TARGET+'.pickle', 'wb') as handle:
    pickle.dump(Data_Test, handle, protocol=pickle.HIGHEST_PROTOCOL)








