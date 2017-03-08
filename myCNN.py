# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 11:21:36 2017

@author: HGY
"""

import h5py
import numpy as np


DATA_TRAIN_PATH =  '../features/Spec/train_sample/' 
DATA_TEST_PATH = '../features/Spec/devel/' 
f = h5py.File(DATA_TEST_PATH+'devel_0001.wav.h5', 'r')
features = f['features'].value
time = f['times'].value