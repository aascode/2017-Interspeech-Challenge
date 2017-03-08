# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:35:42 2017

@author: HGY
"""

import os

## ----------------------------  Parameters  --------------------------------
SR = 22050  #sampling rate
FPS = 70    #framge per sec
FFTLEN= 1024  
FMIN = 50
FMAX = 11000
BANDS = 80
WAV_TRAIN_ROOT = ('../../wav/train_sample/')
WAV_DEVEL_ROOT = ('../../wav/devel_sample/')
SPEC_OUT_TRAIN = ('../../features/Spectrogram/train_sample/')
SPEC_OUT_DEVEL = ('../../features/Spectrogram/devel_sample/')


## ----------------------------  Generate spectrogram feature as .h5  <Train_sample>--------------------------------
TrainList = []
for root, dirs, files in os.walk(WAV_TRAIN_ROOT):
    for file in files:
        if file.endswith(".wav"):
             TrainList.append(os.path.split(file)[1])
TrainList = sorted(TrainList)

for fileName in TrainList:
    print fileName
    extract_melspect = ('python extract_melspect.py --channels=mix-after -r '+str(SR)+' -f '+str(FPS)+' -l '+str(FFTLEN)+' -t mel -m '+str(FMIN)+' -M '+str(FMAX)+' \
                        -b '+str(BANDS)+' -s log --featname "features" --include-times --times-mode=borders \
                        '+WAV_TRAIN_ROOT+fileName+' '+SPEC_OUT_TRAIN+fileName.split('.')[0]+'.h5')
    os.system(extract_melspect)


## ----------------------------  Generate spectrogram feature as .h5 <Devel_sample> --------------------------------
Devel_List = []
for root, dirs, files in os.walk(WAV_DEVEL_ROOT):
    for file in files:
        if file.endswith(".wav"):
             Devel_List.append(os.path.split(file)[1])
Devel_List = sorted(Devel_List)

for fileName in Devel_List:
    print fileName
    extract_melspect = ('python extract_melspect.py --channels=mix-after -r '+str(SR)+' -f '+str(FPS)+' -l '+str(FFTLEN)+' -t mel -m '+str(FMIN)+' -M '+str(FMAX)+' \
                        -b '+str(BANDS)+' -s log --featname "features" --include-times --times-mode=borders \
                        '+WAV_DEVEL_ROOT+fileName+' '+SPEC_OUT_DEVEL+fileName.split('.')[0]+'.h5')
    os.system(extract_melspect)
