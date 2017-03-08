# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:31:29 2017

@author: HGY
"""

'''
C:1 / NC:0
'''

import os
import pickle

# Load label file
LABEL_PATH = '../lab/ComParE2017_Cold.tsv'
OUT_PATH = '../lab/'
TRAIN_TARGET = 'train_final_small2'
TEST_TARGET = 'test_final_small2'
TRAIN_EX = 'devel'


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


# Read sample train list
TRAIN_PATH = ('../wav/'+TRAIN_TARGET+'/')
TrainList = []
for root, dirs, files in os.walk(TRAIN_PATH):
    for file in files:
        if file.endswith(".wav"):
             TrainList.append(os.path.split(file)[1])
labTrain = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in TrainList if (key.split('.')[0] in label_dict) and (key.split('.')[0].split('_')[0] != TRAIN_EX) }
with open('../lab/label_'+TRAIN_TARGET+'.pickle', 'wb') as handle:
    pickle.dump(labTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# Read sample test list
DEVEL_PATH = ('../wav/'+TEST_TARGET+'/')
DevelList = []
for root, dirs, files in os.walk(DEVEL_PATH):
    for file in files:
        if file.endswith(".wav"):
             DevelList.append(os.path.split(file)[1])
labDevel = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in DevelList}
with open('../lab/label_'+TEST_TARGET+'.pickle', 'wb') as handle:
    pickle.dump(labDevel, handle, protocol=pickle.HIGHEST_PROTOCOL)   


