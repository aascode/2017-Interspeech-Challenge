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
TRAIN_PATH = ('../wav/train_sample/')
TrainList = []
for root, dirs, files in os.walk(TRAIN_PATH):
    for file in files:
        if file.endswith(".wav"):
             TrainList.append(os.path.split(file)[1])
labTrain = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in TrainList}

    
# Read sample devel list
DEVEL_PATH = ('../wav/devel_sample/')
DevelList = []
for root, dirs, files in os.walk(DEVEL_PATH):
    for file in files:
        if file.endswith(".wav"):
             DevelList.append(os.path.split(file)[1])
labDevel = {key.split('.')[0]:label_dict[key.split('.')[0]] for key in DevelList}


# Output as pickle
with open('../lab/label_train.pickle', 'wb') as handle:
    pickle.dump(labTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../lab/label_devel.pickle', 'wb') as handle:
    pickle.dump(labDevel, handle, protocol=pickle.HIGHEST_PROTOCOL)   


