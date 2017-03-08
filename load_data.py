# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:00:27 2017

@author: HGY
"""

import pickle
import numpy as np

FEA_ROOT = '../features/ComPare_2016/LLD/'
LAB_ROOT = '../lab/'


# Laod data
TARGET = 'train_final_small2'
with open(FEA_ROOT+'lld_'+TARGET+'.pickle', 'rb') as handle:
    Data_Train = pickle.load(handle)
with open(LAB_ROOT+'label_'+TARGET+'.pickle', 'rb') as handle:
    Label_Train = pickle.load(handle)

TARGET = 'test_final_small2'
with open(FEA_ROOT+'lld_'+TARGET+'.pickle', 'rb') as handle:
    Data_Test = pickle.load(handle)
with open(LAB_ROOT+'label_'+TARGET+'.pickle', 'rb') as handle:
    Label_Test = pickle.load(handle)


''' 
# Concat data to matrix for building GMM
DIM = Data_Train.itervalues().next().shape[1]
Data_mat = np.empty([0, DIM])
for index, (key, value) in enumerate(sorted(Data_Train.items())):
    Data_mat = np.concatenate([Data_mat, value], axis=0)
    print index, key, Label_Train[key]
del Data
'''
'''
### ----------------------------  Fisher Encodig --------------------------------
FV_data = pd.DataFrame()
for index, (key, value) in enumerate(sorted(Label_Train.items())):
    fv = FisherEncode(value, GMMmodel['mean'], GMMmodel['sigma'], GMMmodel['prior'], improved = False, fast=False)
    fv_ = pd.DataFrame(np.concatenate([fv, np.asmatrix(Label[key])], axis=1), index=[key])
    FV_data = pd.concat([FV_data,fv_], axis=0)
    print index, key, Label[key]
FV_data.to_pickle(GMM_ROOT+'FV_'+GMM_TARGET+'_'+TARGET+'_m'+str(MIXURE)+'.pickle')  # where to save it, usually as a .pkl
'''