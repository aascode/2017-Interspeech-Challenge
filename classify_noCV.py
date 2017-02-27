# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:54:57 2017

@author: HGY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 12:25:13 2017
Classifier for functional mat input
@author: HGY
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import scipy.io
from sklearn.metrics import confusion_matrix


FEA_ROOT = '../features/ComPare_2016/'
MIXURE = 64
CLASS_WEIGHT = 'balanced'
KERNEL = 'linear'

### ----------------------------  Load traiing data & devel data (fisher-encoded)  --------------------------------
# Laod train Labels
with open('../lab/label_train.pickle', 'rb') as handle:
    content = pickle.load(handle)
content = sorted(content.items())
Label_train= np.array([x[1] for x in content])

# Load train data
mat = scipy.io.loadmat(FEA_ROOT+'FV_train_m'+str(MIXURE)+'.mat')
Data_train = mat['FV_train']
Data_train = Data_train[:,1:]

# Laod devel Labels
with open('../lab/label_devel.pickle', 'rb') as handle:
    content = pickle.load(handle)
content = sorted(content.items())
Label_devel= np.array([x[1] for x in content])

# Load devel data
mat = scipy.io.loadmat(FEA_ROOT+'FV_devel_m'+str(MIXURE)+'.mat')
Data_devel = mat['FV_devel']
Data_devel = Data_devel[:,1:]


##--------------------  Cross Validation  --------------------
OutputReport_noCV = pd.DataFrame(columns=('Devel_UAR','Devel_Accuracy','Coeff'))

VALID = 3  # Number of folds to cross validation
Cs = [10]
Percentile = range(10,110,10)



print 'Start Cross Validation...'
for c in Cs: 
    for per in Percentile:
        
        ## ---- Use model built from trainig data to apply on developing data  ----
        # Select percentile
        fs = SelectPercentile(score_func=f_classif,percentile=per).fit(Data_train,Label_train)            
        Data_train_FS = fs.transform(Data_train)
        Data_devel_FS = fs.transform(Data_devel)
        
        # Modeling
        clf = SVC(kernel=KERNEL, C=c, class_weight=CLASS_WEIGHT, verbose=True)
        clf.fit(Data_train_FS, Label_train)
        predict_devel = clf.predict(Data_devel_FS)
        
        
        # Evaluate the model
        uar_devel = recall_score(Label_devel, predict_devel, average='macro')     
        accu_devel = accuracy_score(Label_devel, predict_devel)
        df = pd.DataFrame({'Devel_UAR':str(round(uar_devel,3)), 'Devel_Accuracy':str(round(accu_devel,3)), 'Coeff':'C:'+str(c)+' Mixure: '+str(MIXURE)}, index=[per])
        OutputReport_noCV = pd.concat([OutputReport_noCV,df])


# Output to xls for record
OutputReport_noCV = OutputReport_noCV[['Devel_UAR','Devel_Accuracy','Coeff']]
OutputReport_noCV.to_excel('./result/FV_noCV_'+str(MIXURE)+'.xlsx',index=True, header=True)
print('Done')
#
#
