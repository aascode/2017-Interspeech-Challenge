# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:21:35 2017

@author: HGY
"""

import os
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
            C_dict[key] = 1
            All_dict[key] = 1      
        else:
            NC_dict[key] = 0 
            All_dict[key] = 0   
                   

# get train list
AUDIO_PATH_OLD = ('../wav/train_balance5/')
TrainDict = {}
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for f in files:
        if f.endswith(".wav"):
             #print(os.path.join(root, file))
             TrainDict[f.split('.')[0]] = os.path.abspath(f)

# get rest for test
Test_large = {k:All_dict[k] for k in All_dict if k not in TrainDict}
 
# Move
AUDIO_PATH_OLD = ('../wav/train_devel/')
AUDIO_PATH_NEW = ('../wav/test_balance5_large/')
for idx, (key,val) in enumerate(Test_large.items()):
    print idx, key
    fileName = key+'.wav'
    copyfile(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW,fileName))


