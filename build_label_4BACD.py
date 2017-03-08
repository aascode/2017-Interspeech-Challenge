# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 15:18:47 2017

@author: HGY
"""

import os
import pandas as pd

# Load label file
LABEL_PATH = '../lab/ComParE2017_Cold.tsv'
OUT_PATH = '../lab/'
with open(LABEL_PATH) as f:
    content = f.readlines()
INFO = content[0]
content = content[1:]


# Build dictionary
label_dict = {}
for lab in content:
    lab = lab.replace('\n','').split('\t')
    key = lab[0].split('.')[0]
    label_dict[key] = 1 if lab[1]=='C' else 0
'''
## ----------------------------  Sample Train  --------------------------------
TARGET = 'train_final'
WAV_PATH = '../wav/for CNN/'+TARGET+'/'
DataList = []
for root, dirs, files in os.walk(WAV_PATH):
    for file in files:
        if file.endswith(".wav"):
             DataList.append(os.path.split(file)[1])
labels = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in DataList}
tmp = pd.DataFrame.from_dict({'label':labels}, orient='columns',  dtype=None)
tmp.to_csv(OUT_PATH+TARGET+'.csv',index=True, header=True)
'''

TARGET = 'test_final'
WAV_PATH = '../wav/for CNN/'+TARGET+'/'
DataList = []
for root, dirs, files in os.walk(WAV_PATH):
    for file in files:
        if file.endswith(".wav"):
             DataList.append(os.path.split(file)[1])
labels = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in DataList}
tmp = pd.DataFrame.from_dict({'label':labels}, orient='columns',  dtype=None)
tmp.to_csv(OUT_PATH+TARGET+'.csv',index=True, header=True)

