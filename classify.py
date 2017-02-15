# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 12:25:13 2017
Classifier for functional mat input
@author: HGY
"""

from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt


##--------------------  Variable  Settings  --------------------
VALID = 10  # Number of folds to cross validation
c = 1
Percentile = range(10,110,10)

# Import from .mat fisher vector
mat = scipy.io.loadmat('../features2/1functional_Inter_nodiv.mat')
feaData = mat['HrAudFunc']
audioIdx = feaData[:,0]
feaData = feaData[:,1:]
feaData = pd.DataFrame(feaData,index=audioIdx)
#mat = scipy.io.loadmat('../features/iHR_Label_0.mat')
#label = pd.DataFrame(mat['Data'])

# Import label
labLoc = '../features2/label_Inter_1functional.xlsx'
labels = pd.read_excel(labLoc, 0, index_col=0, header=0, na_values=['Nan'])   
labels = labels.dropna(how='any')
print 'Greater:',len(labels[(labels>0)].dropna()), 'Unchange:',len(labels[(labels==0)].dropna())


# Fetch index 
with pd.option_context('mode.use_inf_as_null', True):
   feaData = feaData.dropna(how='any')
#feaData = feaData.dropna(how='any')
audioIdx = pd.Index(feaData.index)
labIdx = pd.Index(labels.index)
allIdx = labIdx.intersection(audioIdx)
feaData = feaData.ix[allIdx]
labels = labels.ix[allIdx]
feaData = np.asmatrix(feaData)
labels = np.asmatrix(labels)

TestSize = int(len(labIdx)/VALID)
TrainSize = len(labIdx)-TestSize
Order = np.random.permutation(len(labels))

print 'Start Cross Validation...'
outputReport = pd.DataFrame(columns=('Spearman', 'Pearson', 'Rmse' , 'Coeff', 'UAR'))
for per in Percentile:    # SELECT PERCENTILE 
    Rmses = []
    Spears = []
    Pears = []
    Uars = []
    Predicts = []    
    GroundTruths = []
    
    uar = []
    for VAL in range(VALID):
        print VAL, per
        testIdx = Order[VAL*TestSize:(VAL+1)*TestSize]
        mask = np.in1d(Order, testIdx)
        trainIdx = Order[~mask]
               
        trainFea = feaData[trainIdx,:]
        trainLab = labels[trainIdx,:]
        testFea = feaData[testIdx,:]
        testLab = labels[testIdx,:]
        GroundTruths = GroundTruths+np.ravel(testLab).tolist()
        
        # Training and predicting using SVR 
        clf = SVC(kernel='linear', C=c,class_weight='balanced') # LINEAR SVR
        

        # DO FEATURE SELECTION
        fs = SelectPercentile(score_func=f_regression,percentile=per).fit(trainFea,np.ravel(trainLab))            
        trainFea_FS = fs.transform(trainFea)
        testFea_FS = fs.transform(testFea)        
        fsVal = fs.get_support(indices=True)
                
        # Predicting
        clf.fit(trainFea_FS, np.ravel(trainLab))   
        predict = clf.predict(testFea_FS)
        Predicts = Predicts+predict.tolist()
               
    # Evaluation        
    conf = confusion_matrix(GroundTruths,Predicts, labels=[0,1])
    Rmse = sqrt(mean_squared_error(GroundTruths, Predicts))
    Spear = spearmanr(Predicts, GroundTruths)[0]
    Pear = pearsonr(Predicts, GroundTruths)[0]
    Uar = recall_score(GroundTruths, Predicts, average='macro')     
    df = pd.DataFrame({'Spearman':str(round(Spear,3)), 'Pearson':str(round(Pear,3)), 'Rmse':str(round(Rmse,3)), 'Coeff':str(c), 'UAR':str(round(Uar,3))}, index=[per])
    outputReport = pd.concat([outputReport,df])


# Output to xls for record
#outputReport = outputReport[['Spearman', 'Pearson', 'Rmse' , 'Coeff', 'UAR']]
#outputReport.to_excel('../result2/1functional_Inter_prev_nodiv.xlsx',index=True, header=True)
print('Done')


