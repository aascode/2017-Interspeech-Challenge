# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 22:15:18 2017

@author: HGY
"""

import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats


FEA_ROOT = '../features/ComPare_2016/FV/'
GMM = 'train_balance5'
TRAIN_TARGET = 'train_balance5'
TEST_TARGET = 'test_balance5'
MIXURE = 128
CLASS_WEIGHT = 'balanced'
KERNEL = 'linear'
ZNORM = False
SIGNORM = False
L2NORM = False
OUT_NAME = './result/FV_'+GMM+'_'+str(MIXURE)


### ----------------------------  Load traiing data & test data (fisher-encoded)  --------------------------------
# Laod data 
with open(FEA_ROOT+'FV_'+GMM+'_'+TRAIN_TARGET+'_m'+str(MIXURE)+'.pickle', 'rb') as handle:
    Data_train = pickle.load(handle)
with open(FEA_ROOT+'FV_'+GMM+'_'+TEST_TARGET+'_m'+str(MIXURE)+'.pickle', 'rb') as handle:
    Data_test = pickle.load(handle)
                                                                
Data_train = Data_train.as_matrix()
Data_test = Data_test.as_matrix()

Label_train = Data_train[:,-1]
Label_test = Data_test[:,-1]
Data_train = Data_train[:,:-1]
Data_test = Data_test[:,:-1]

##--------------------  Fisher Vector Normalizations  --------------------
def FISHER_NORM(Data_train, Data_test, ZNORM, SIGNORM, L2NORM, OUT_NAME):
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    def sigmoid2D(x):
        return map(sigmoid,x)  
    
    if ZNORM:
        print 'Apply feature-level Z-scrore normalization!'
        Data_train = stats.zscore(Data_train, axis=0)
        Data_test = stats.zscore(Data_test, axis=0)
        Data_train = np.nan_to_num(Data_train) #prevent nan values
        Data_test = np.nan_to_num(Data_test)    
        OUT_NAME = OUT_NAME+'_Znorm'

    if SIGNORM:
        print 'Apply element-wise sigmoid normalization!'
        Data_train = np.apply_along_axis(sigmoid2D, 0, Data_train)
        Data_test = np.apply_along_axis(sigmoid2D, 0, Data_test)
        Data_train = np.nan_to_num(Data_train) #prevent nan values
        Data_test = np.nan_to_num(Data_test)    
        OUT_NAME = OUT_NAME+'_Signorm'

    if L2NORM:
        print 'Apply instance-level L2 normalizaton!'
        Data_train = preprocessing.normalize(Data_train, norm='l2', axis=1)
        Data_test = preprocessing.normalize(Data_test, norm='l2', axis=1)
        OUT_NAME = OUT_NAME+'_L2norm'

    return Data_train, Data_test, OUT_NAME
Data_train, Data_test, OUT_NAME = FISHER_NORM(Data_train, Data_test, ZNORM, SIGNORM, L2NORM, OUT_NAME)



##--------------------  Modeling  --------------------
OutputReport_noCV = pd.DataFrame(columns=('Test_UAR','Test_Accuracy','Coeff'))
Cs = [0.1,1,10]
Percentile = range(10,110,10)


print 'Start Cross Validation...'
for c in Cs: 
    for per in Percentile:
        
        ## ---- Use model built from trainig data to apply on testing data  ----
        # Select percentile
        fs = SelectPercentile(score_func=f_classif,percentile=per).fit(Data_train,Label_train)            
        Data_train_FS = fs.transform(Data_train)
        Data_test_FS = fs.transform(Data_test)
        
        # Modeling
        clf = SVC(kernel=KERNEL, C=c, class_weight=CLASS_WEIGHT, verbose=True)
        clf.fit(Data_train_FS, Label_train)
        predict_test = clf.predict(Data_test_FS)
        
        # Evaluate the model
        uar_test = recall_score(Label_test, predict_test, average='macro')     
        accu_test = accuracy_score(Label_test, predict_test)
        df = pd.DataFrame({'Test_UAR':round(uar_test,3), 'Test_Accuracy':round(accu_test,3), 'Coeff':'C:'+str(c)+' Mixure: '+str(MIXURE)}, index=[per])
        OutputReport_noCV = pd.concat([OutputReport_noCV,df])


# Output to xls for record
OutputReport_noCV = OutputReport_noCV[['Test_UAR','Test_Accuracy','Coeff']]
OutputReport_noCV.to_excel(OUT_NAME+'.xlsx',index=True, header=True)
print('Done')
#
#
