# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 21:10:25 2017

@author: HGY
"""

import pickle
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


LAB_ROOT = '../lab/'
GROUND_TRUTH = 'devel_sample'
TARGET = 'balance3'
THRESHOLD = 0.3


# Laod prediction
Predict_final = pd.read_csv('../CNN/'+TARGET+'/prediction_final.csv',index_col=0).sort_index()
Predict_first = pd.read_csv('../CNN/'+TARGET+'/prediction_first.csv',index_col=0).sort_index()
Predict_second = pd.read_csv('../CNN/'+TARGET+'/prediction_second.csv',index_col=0).sort_index()

# Load gound truth
with open(LAB_ROOT+'label_'+GROUND_TRUTH+'.pickle', 'rb') as handle:
    GroundTruth = pickle.load(handle)
GroundTruth = pd.DataFrame.from_dict(GroundTruth,orient='index').sort_index()


# classify the result to 0/1
Predict_final[Predict_final>=THRESHOLD] = 1
Predict_final[Predict_final<THRESHOLD] = 0
Predict_first[Predict_first>=THRESHOLD] = 1
Predict_first[Predict_first<THRESHOLD] = 0
Predict_second[Predict_second>=THRESHOLD] = 1
Predict_second[Predict_second<THRESHOLD] = 0


# Scoring
uar_final = recall_score(GroundTruth.as_matrix(), Predict_final.as_matrix(), average='macro')     
accu_final = accuracy_score(GroundTruth.as_matrix(), Predict_final.as_matrix())
uar_first = recall_score(GroundTruth.as_matrix(), Predict_first.as_matrix(), average='macro')     
accu_first = accuracy_score(GroundTruth.as_matrix(), Predict_first.as_matrix())
uar_second = recall_score(GroundTruth.as_matrix(), Predict_second.as_matrix(), average='macro')     
accu_second = accuracy_score(GroundTruth.as_matrix(), Predict_second.as_matrix())
print 'Final:','uar:',uar_final,'accu:',accu_final
print 'First:','uar:',uar_first,'accu:',accu_first
print 'Second:','uar:',uar_second,'accu:',accu_second





