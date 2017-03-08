# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 19:29:59 2017

@author: HGY
"""

import pickle
import numpy as np
from sklearn import mixture

TARGET = 'train_final_small'
LLD_ROOT = '../features/ComPare_2016/LLD/'
LAB_ROOT = '../lab/'
OUT_ROOT = '../features/ComPare_2016/FV/'
MIXURE = [4,8,16,32,64,128,256]

### ----------------------------  Load traiing data & label --------------------------------
# Laod train label
with open(LAB_ROOT+'label_'+TARGET+'.pickle', 'rb') as handle:
    Label_train = pickle.load(handle)
# Laod train data
with open(LLD_ROOT+'/lld_'+TARGET+'.pickle', 'rb') as handle:
    Data = pickle.load(handle)


### ----------------------------  Concat data for building GMM --------------------------------
DIM = Data.itervalues().next().shape[1]
Data_mat = np.empty([0, DIM])
for index, (key, value) in enumerate(sorted(Data.items())):
    Data_mat = np.concatenate([Data_mat, value], axis=0)
    print index, key
del Data
    

### ----------------------------  Build GMM by EM --------------------------------
for mix in MIXURE:
    print 'Now build gmm with mixure:', mix
    gmmModel = {}
    gmm = mixture.GaussianMixture(n_components=mix, covariance_type='diag').fit(Data_mat)
    gmmModel['prior'] = gmm.weights_
    gmmModel['mean'] = gmm.means_
    gmmModel['sigma'] = gmm.covariances_

    # Output as pickle
    with open(OUT_ROOT+'GMM_'+TARGET+'_m'+str(mix)+'.pickle', 'wb') as handle:
        pickle.dump(gmmModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    

    

