# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:12:54 2016

@author: Keane-pc
"""
# Import libraries
from __future__ import division
import numpy as np
import librosa  # Librosa for audio
import pandas as pd
from scipy import stats
from sklearn import cluster

## load file and do zscore
"""
audio_path = 'E:/Summer/NTUA_2016_07_04/angry/SPLITED_Audio/01/A/split0009.wav'
track,sr = librosa.load(audio_path, sr=16000)  # sampling rate of sound is 16000
track = stats.zscore(track)
"""

# Generate Location of files 
Location = []
lay0 = range(1,23)
lay1 = ['/A/', '/B/']
lay2 = ['angry', 'happy', 'neutral', 'sad', 'frustration', 'surprise']
for team in lay0:
    if team<10:
        team = '0' + str(team)
    else:
        team = str(team)
    for actor in lay1:
        for emo in lay2:
            Location.append('../data/' + emo + '/SPLITED_Audio/' + team + actor)


# load all emotion soundtracks of an actor
emoSize = len(lay2)
features = []
allM = []
trackCounts = []
frameCounts = []
hop_length = 256

allData = []
for j in range(0,44):

    personData = []
    for i in range(j*6,j*6+emoSize):
        track = []
        print Location[i]
        files = librosa.util.find_files(Location[i], ext='wav', case_sensitive=True)
        trackCounts.append(len(files))
        
        if not files:
            print Location[i] + " is garbage = ="
            frameCounts.append(0)
            continue

        sessionData = []        
        for filedir in files:
            sound,sr = librosa.load(filedir, sr=16000)  # sampling rate of sound is 16000

            ## Extract MFCC, delMFCC, del2MFCC from track
            S = librosa.feature.melspectrogram(sound, sr=sr, n_mels=128,hop_length=hop_length)
            log_S = librosa.logamplitude(S, ref_power=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            delta_mfcc  = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            soundFeature = np.vstack([mfcc, delta_mfcc, delta2_mfcc])        
            frameCounts.append(soundFeature.shape[1])
            sessionData = sessionData + soundFeature.T.tolist()
        
        personData = personData + sessionData
    
    # Do zscore along each rows (each features), then transpose for KMeans input
    personData = np.asarray(personData)    
    personData = stats.zscore(personData,axis=0)
    allData = allData + personData.tolist()
    
    
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


