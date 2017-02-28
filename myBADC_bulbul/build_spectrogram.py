# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:35:42 2017

@author: HGY
"""

import os
import random

## ----------------------------  Parameters  --------------------------------
SR = 22050  #sampling rate
FPS = 70    #framge per sec
FFTLEN= 1024  
FMIN = 50
FMAX = 11000
BANDS = 80
WAV_TRAIN_ROOT = ('../../wav/train_sample/')
WAV_DEST_ROOT = ('../../wav/devel_sample/')



## ----------------------------  Constants  --------------------------------
TrainList = []
for root, dirs, files in os.walk(WAV_TRAIN_ROOT):
    for file in files:
        if file.endswith(".wav"):
             TrainList.append(os.path.split(file)[1])
        
path = os.path.abspath(TrainList[0])

extract_melspect.py --channels=mix-after -r ${SR} -f ${FPS} -l ${FFTLEN} -t mel -m ${FMIN} -M ${FMAX} -b ${BANDS} -s log --featname "features" --include-times --times-mode=borders "$f" "$o"; then

                                             
extract_melspect.py --channels=mix-after -r 22050 -f 70 -l 1024 -t mel -m 50 -M 11000 -b 80 -s log --featname "features" --include-times --times-mode=borders ../../wav/train_sample/train_0008.wav ../../features/SPEC/train_008.h5

