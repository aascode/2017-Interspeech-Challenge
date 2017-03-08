# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 10:08:27 2017

@author: HGY
"""

import os
import random
from shutil import copyfile


## ----------------------------  Load all label  --------------------------------
# Load label file
LABEL_PATH = '../lab/ComParE2017_Cold.tsv'
OUT_PATH = '../lab/'
with open(LABEL_PATH) as f:
    content = f.readlines()
INFO = content[0]
content = content[1:]

# Build dictionary
C_dict = {}
NC_dict = {}
All_dict = {}
for lab in content:
    lab = lab.replace('\n','').split('\t')
    key = lab[0].split('.')[0]
    if (key.split('_')[0]) =='train' or 'devel':
        if lab[1]=='C':
            All_dict[key] = 1
            C_dict[key] = 1
        else:
            All_dict[key] = 0
            NC_dict[key] = 0    

## ----------------------------  for training data  --------------------------------
# Move NC .wav
AUDIO_PATH_OLD = ('../wav/train/')
TrainDict = {}
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for f in files:
        if f.endswith(".wav"):
             TrainDict[f.split('.')[0]] = os.path.abspath(f)
       
        
C_dict_train = dict((k, C_dict[k]) for k in TrainDict if k in C_dict)
NC_dict_train = dict((k, NC_dict[k]) for k in TrainDict if k in NC_dict)
SAMPLE_SIZE = len(C_dict_train.keys())
random_NC = random.sample(NC_dict_train.items(),SAMPLE_SIZE)

# Move 1000 C & 1000NC to train 
AUDIO_PATH_TRAIN = ('../wav/train_final_small2/')
for key in sorted(random_NC):
    print 'NC',key[0]
    fileName = key[0]+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_TRAIN,fileName))
for key in sorted(C_dict_train):
    print 'C',key
    fileName = key+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_TRAIN,fileName))





## ----------------------------  for testing data  --------------------------------
# Move 1000 C & 1000NC to train 
AUDIO_PATH_OLD = ('../wav/devel/')
AUDIO_PATH_TEST = ('../wav/test_final_small2/')
DevelDict = {}
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for f in files:
        if f.endswith(".wav"):
             DevelDict[f.split('.')[0]] = os.path.abspath(f)


C_dict_test = dict((k, C_dict[k]) for k in DevelDict if k in C_dict)
NC_dict_test = dict((k, NC_dict[k]) for k in DevelDict if k in NC_dict)
SAMPLE_SIZE = len(C_dict_test.keys())
random_NC = random.sample(NC_dict_test.items(),SAMPLE_SIZE)

for key in sorted(random_NC):
    print 'NC',key[0]
    fileName = key[0]+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_TEST,fileName))

for key in sorted(C_dict_test):
    print 'C',key
    fileName = key+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_TEST,fileName))
 
    

# sample additional 1000 devel for training data
random_test = random.sample(DevelDict.items(),SAMPLE_SIZE)
for (key,val) in random_test:   
    print key, All_dict[key]
    fileName = key+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_TRAIN,fileName))

