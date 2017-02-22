# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:55:05 2017

@author: HGY
"""

import os
import random

SAMPLE_SIZE = 1000


## ----------------------------  Train Data  --------------------------------
# Move Files
SAMPLE_SEED_TRAIN = 1
AUDIO_PATH_OLD = ('../wav/train/')
AUDIO_PATH_NEW = ('../wav/train_sample/')

TrainList = []
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for file in files:
        if file.endswith(".wav"):
             #print(os.path.join(root, file))
             TrainList.append(os.path.abspath(file))
             
random.seed(a=SAMPLE_SEED_TRAIN)
sampTrain = [ TrainList[i] for i in sorted(random.sample(xrange(len(TrainList)), SAMPLE_SIZE))]
for i in range(len(sampTrain)):
    fileName = os.path.split(sampTrain[i])[1]
    os.rename(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW, fileName))



## ----------------------------  Devel Data  --------------------------------
# Move Files
SAMPLE_SEED_DEVEl = 2
AUDIO_PATH_OLD = ('../wav/devel/')
AUDIO_PATH_NEW = ('../wav/devel_sample/')

DevelList = []
for root, dirs, files in os.walk(AUDIO_PATH_OLD):
    for file in files:
        if file.endswith(".wav"):
             #print(os.path.join(root, file))
             DevelList.append(os.path.abspath(file))
             
random.seed(a=SAMPLE_SEED_DEVEl)
sampDevel = [ DevelList[i] for i in sorted(random.sample(xrange(len(DevelList)), SAMPLE_SIZE))]
for i in range(len(sampDevel)):
    fileName = os.path.split(sampDevel[i])[1]
    os.rename(os.path.join(AUDIO_PATH_OLD,fileName), os.path.join(AUDIO_PATH_NEW, fileName))