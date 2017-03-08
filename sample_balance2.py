# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 22:23:56 2017

@author: HGY
"""

import os
import random
import pickle
import pandas as pd


LABEL_PATH = '../lab/ComParE2017_Cold.tsv'
LLD_PATH = '../features/ComPare_2016/LLD/'
LABEL_PATH = '../lab/ComParE2017_Cold.tsv'

## ----------------------------  Load all label  --------------------------------
with open(LABEL_PATH) as f:
    content = f.readlines()
INFO = content[0]
content = content[1:]
label_dict = {}
for lab in content:
    lab = lab.replace('\n','').split('\t')
    key = lab[0].split('.')[0]
    label_dict[key] = 1 if lab[1]=='C' else 0
              

# Build dictionary
C_dict = {}
NC_dict = {}
for lab in content:
    lab = lab.replace('\n','').split('\t')
    key = lab[0].split('.')[0]
    if (key.split('_')[0]) =='train' or 'devel':
        if lab[1]=='C':
            C_dict[key] = 1
        else:
            NC_dict[key] = 0    


## ----------------------------  Load lld dict  --------------------------------
# Load lld index dict
with open(LLD_PATH+'LLD_Index_train.pickle', 'rb') as handle:
    Idx_train = pickle.load(handle)
with open(LLD_PATH+'LLD_Index_devel.pickle', 'rb') as handle:
    Idx_devel = pickle.load(handle)
Idx_train.update(Idx_devel)
Idx_all = Idx_train
del Idx_train, Idx_devel

## ----------------------------  Sample & Move train data  --------------------------------
Train_Data = {}
Test_Data = {}
Train_Label = {}
Test_Label = {}

# Add cold data
random_C = random.sample(C_dict.items(),len(C_dict))
for idx, (key,val) in enumerate(random_C):
    print idx, key
    # get start point and nrows
    st = Idx_all[key][0]
    N = Idx_all[key][1]
    
    # get feature
    if key.split('_')[0] == 'train':
        fea = pd.read_table(LLD_PATH+'train.csv', delimiter=';', nrows=N, skiprows=st)
    else:
        fea = pd.read_table(LLD_PATH+'devel.csv', delimiter=';', nrows=N, skiprows=st)    
    
    fea = fea.as_matrix()
    fea = fea[:,1:].astype('float')
    
    if idx < len(random_C)/2:
        Train_Data[key] = fea
        Train_Label[key] = label_dict[key] 
    else:
        Test_Data[key] = fea
        Test_Label[key] = label_dict[key] 
        

















# Move NC .wav
AUDIO_PATH_OLD = ('../wav/train_devel/')
AUDIO_PATH_NEW = ('../wav/train_balance4/')

TrainDict = {}
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for f in files:
        if f.endswith(".wav"):
             #print(os.path.join(root, file))
             TrainDict[f.split('.')[0]] = os.path.abspath(f)
       
        
C_dict = dict((k, C_dict[k]) for k in TrainDict if k in C_dict)
NC_dict = dict((k, NC_dict[k]) for k in TrainDict if k in NC_dict)
SAMPLE_SIZE = len(C_dict.keys())


random_NC = random.sample(NC_dict.items(),SAMPLE_SIZE)






