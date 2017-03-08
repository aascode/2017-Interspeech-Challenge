# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:06:07 2017

@author: JohnLi
"""

import numpy as np
import pandas as pd
import pickle

TARGET = 'devel'
FEA_ROOT = '../features/ComPare_2016/LLD/'


# load the first column (ex) "train_0003.wav"
Idxs = pd.read_table(FEA_ROOT+TARGET+'.csv', delimiter=';', usecols=[0])

# keep unique wavnames
Wav_names = Idxs['name'].drop_duplicates()

# specify the starting Idxs and number of samples for each wav
Start_point = Wav_names.axes[0].tolist()
Nsamples = np.append(np.diff(Start_point), len(Idxs)-Start_point[-1])

Wav_names = Wav_names.tolist()
Idx_dict = {}
for ii in range(len(Nsamples)):
    Idx_dict[Wav_names[ii].replace('\'','').split('.')[0]] = [Start_point[ii],Nsamples[ii]]


# Output as pickle
with open(FEA_ROOT+'LLD_Index_'+TARGET+'.pickle', 'wb') as handle:
    pickle.dump(Idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




'''
for st, N in zip(Start_point, Nsamples): 
    # load the table with specific wav 
    # skip the previous rows of other waves and load the specified nrows
    feature = pd.read_table(PATH, delimiter=';', nrows=N, skiprows=st)
    feature = feature.iloc[:,1:].as_matrix()
'''