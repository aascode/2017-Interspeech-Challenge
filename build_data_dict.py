# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:27:27 2017

@author: HGY
"""

import csv
import numpy as np
import numpy.matlib
import pickle
from scipy.io import savemat
from scipy.io import loadmat


FEA_ROOT = '../features/ComPare_2016/'


### ----------------------------  Train Data  --------------------------------
# Laod Labels to initialize data dict
with open('../lab/label_train.pickle', 'rb') as handle:
    Label_train = pickle.load(handle)

# build lld feature dict
DATA_TRAIN_PATH = FEA_ROOT+'train_sample.csv'
Data_Train = {k:[] for k in Label_train.keys()}

f = open(DATA_TRAIN_PATH, 'r')  
reader = csv.reader(f)
headers = reader.next()
FeaName = ';'.join(headers).split(';')
print 'There dimension of the feature is',len(FeaName)-1

for row in reader:  
    r = ';'.join(row).split(';')
    key = r[0].replace('\'','').split('.')[0]
    val = np.array(r[1:]).astype('float')
    Data_Train[key].append(val)
    print key
f.close()
Data_Train = dict(map(lambda (k,v): (k, np.array(v)), Data_Train.iteritems()))

# Output as pickle
with open(FEA_ROOT+'lld_train.pickle', 'wb') as handle:
    pickle.dump(Data_Train, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Output as mat for fisher vector
Train_mat = np.empty([0, len(FeaName)])
for index, (key, value) in enumerate(sorted(Data_Train.items())):
    idx  = np.matlib.repmat(np.array([index]), value.shape[0], 1)
    tmp = np.concatenate([value,idx], axis=1)
    Train_mat = np.concatenate([Train_mat, tmp], axis=0)
    print index, key, Label_train[key]
savemat(FEA_ROOT+'train4FV', mdict={'Data_train': Train_mat}, appendmat=True)

#saEA_ROOT+'train4FV', mdict={'Data_Train':Data_Train}, appendmat=True)




    
### ----------------------------  Devel Data  --------------------------------
# Laod Labels to initialize data dict
with open('../lab/label_devel.pickle', 'rb') as handle:
    Label_devel = pickle.load(handle)

# build lld feature dict
DATA_DEVEL_PATH = FEA_ROOT+'devel_sample.csv'
Data_Devel = {k:[] for k in Label_devel.keys()}

f = open(DATA_DEVEL_PATH, 'r')  
reader = csv.reader(f)
headers = reader.next()
FeaName = ';'.join(headers).split(';')
print 'There dimension of the feature is',len(FeaName)-1

for row in reader:  
    r = ';'.join(row).split(';')
    key = r[0].replace('\'','').split('.')[0]
    val = np.array(r[1:]).astype('float')
    Data_Devel[key].append(val)
    print key
f.close()
Data_Devel = dict(map(lambda (k,v): (k, np.array(v)), Data_Devel.iteritems()))

# Output as pickle
with open(FEA_ROOT+'lld_devel.pickle', 'wb') as handle:
    pickle.dump(Data_Devel, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Output as mat for fisher vector
Devel_mat = np.empty([0, len(FeaName)])
for index, (key, value) in enumerate(sorted(Data_Devel.items())):
    idx  = np.matlib.repmat(np.array([index]), value.shape[0], 1)
    tmp = np.concatenate([value,idx], axis=1)
    Devel_mat = np.concatenate([Devel_mat, tmp], axis=0)
    print index, key, Label_devel[key]
savemat(FEA_ROOT+'devel4FV', mdict={'Data_devel': Devel_mat}, appendmat=True)


#savemat(FEA_ROOT+'devel4FV', mdict={'Data_Devel':Data_Devel}, appendmat=True)








