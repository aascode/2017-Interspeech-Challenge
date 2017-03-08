# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 08:33:08 2017

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
for lab in content:
    lab = lab.replace('\n','').split('\t')
    key = lab[0].split('.')[0]
    if (key.split('_')[0]) =='train' or 'devel':
        if lab[1]=='C':
            C_dict[key] = 1
        else:
            NC_dict[key] = 0    

## ----------------------------  Sample & Move train data  --------------------------------
# Move NC .wav
AUDIO_PATH_OLD = ('../wav/train_devel/')

TrainDict = {}
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for f in files:
        if f.endswith(".wav"):
             #print(os.path.join(root, file))
             TrainDict[f.split('.')[0]] = os.path.abspath(f)
       
        
C_dict = dict((k, C_dict[k]) for k in TrainDict if k in C_dict)
NC_dict = dict((k, NC_dict[k]) for k in TrainDict if k in NC_dict)
SAMPLE_SIZE = len(C_dict.keys())



## ----------------------------  Sample & Move for data 2000  --------------------------------
AUDIO_PATH_NEW = ('../wav/train_fainl_small2/')
AUDIO_PATH_NEW2 = ('../wav/test_final_small2/')

random_NC = random.sample(NC_dict.items(),SAMPLE_SIZE)
random_C = random.sample(C_dict.items(),SAMPLE_SIZE)
for idx in range(len(random_NC)):
    key = random_NC[idx]
    fileName = key[0]+'.wav'
    if idx > len(random_NC)/2:
        print 'train',idx,key[0]
        copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW,fileName))
    else:
        print 'test',idx,key[0]
        copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW2,fileName))

for idx in range(len(random_C)):
    key = random_C[idx]
    fileName = key[0]+'.wav'
    if idx > len(random_C)/2:
        print 'train',idx,key[0]
        copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW,fileName))
    else:
        print 'test',idx,key[0]
        copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW2,fileName))
'''

'''
# Add test data for robustness
TEST_PATH = '../wav/test/'
TestDict = {}
for root, dirs, files in os.walk(TEST_PATH):
    for f in files:
        if f.endswith(".wav"):
             #print(os.path.join(root, file))
             TestDict[f.split('.')[0]] = os.path.abspath(f)
random_Test = random.sample(TestDict.items(),SAMPLE_SIZE/2)
for idx in range(len(random_Test)):
    key = random_Test[idx]
    fileName = key[0]+'.wav'
    print 'train',idx,key[0]
    copyfile(os.path.join(TEST_PATH,fileName), os.path.join(AUDIO_PATH_NEW,fileName))



'''
## ----------------------------  Sample & Move for data 4000 (final)  --------------------------------
AUDIO_PATH_NEW = ('../wav/train_final_large/')

random_NC = random.sample(NC_dict.items(),SAMPLE_SIZE)
random_C = random.sample(C_dict.items(),SAMPLE_SIZE)
for idx in range(len(random_NC)):
    key = random_NC[idx]
    fileName = key[0]+'.wav'
    print 'train',idx,key[0]
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW,fileName))

for idx in range(len(random_C)):
    key = random_C[idx]
    fileName = key[0]+'.wav'
    print 'train',idx,key[0]
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW,fileName))

# Add test data for robustness
TEST_PATH = '../wav/test/'
TestDict = {}
for root, dirs, files in os.walk(TEST_PATH):
    for f in files:
        if f.endswith(".wav"):
             #print(os.path.join(root, file))
             TestDict[f.split('.')[0]] = os.path.abspath(f)
random_Test = random.sample(TestDict.items(),SAMPLE_SIZE)
for idx in range(len(random_Test)):
    key = random_Test[idx]
    fileName = key[0]+'.wav'
    print 'train',idx,key[0]
    copyfile(os.path.join(TEST_PATH,fileName), os.path.join(AUDIO_PATH_NEW,fileName))
'''