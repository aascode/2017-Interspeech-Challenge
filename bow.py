# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:51:07 2017

@author: CHADSHEEP
"""

## Import libraries
from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cluster
import arff


## load file and do zscore
FEATURE_PATH = '../arff/ComParE2017_Cold.ComParE.train.arff'
Featureset = arff.load(open(FEATURE_PATH, 'rb'))
Features = np.array(Featureset['data'])
del Featureset

    
    
## Learn global Kmenas model and predict each samples' class 
k = 8
k_means = cluster.KMeans(n_clusters=k)
clustData = k_means.fit_predict(allData)   


## Split each emotion, calcultate histogram, do zscore, than output 
id1=0
for frameCount in frameCounts:
    emoClus = clustData[id1:id1+frameCount]   
    hist = np.histogram(emoClus,bins=range(0,k+1))
    features.append(hist[0] / float(sum(hist[0])))  
    id1 = id1 + frameCount

labels = []
label = 0
for trackCount in trackCounts:
    if trackCount==0:
        #labels = labels + (np.empty(1) * np.nan).tolist()
        labels = labels + (np.empty(label)).tolist()
        label = label+1
        continue
    labels  = labels + [label]*trackCount
    label = label+1

# Save results to excel
features = np.column_stack((np.asarray(features),np.asanyarray(labels)))
outputFeature = pd.DataFrame(features)

outputFeature.to_excel('./Features.xlsx', index=False, header=False)
print('Done')


