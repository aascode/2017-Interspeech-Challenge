# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 20:42:45 2017

@author: HGY
"""

import pickle
import numpy as np
import pandas as pd

TARGET = 'test_final_small'
GMM_TARGET = 'train_final_small'
LLD_ROOT = '../features/ComPare_2016/LLD/'
GMM_ROOT = '../features/ComPare_2016/FV/'
LAB_ROOT = '../lab/'

MIXURES = [16,32,64,128]

### ----------------------------  Fisher Encoding --------------------------------
def _log_multivariate_normal_density_diag(X, means, covars):
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr
def log_normal_prob(feats, means, covars):
    K, D = means.shape
    lpr = []
    for k in range(K):
        delta = (feats-means[k,:])/(covars[k,:]**0.5)
        lpr.append(- 0.5*sum(np.log(covars[k,:])) - 0.5*np.sum(delta**2,1))
    return np.array(lpr).T

def predict_prob(feats, priors, means, covars):     
    logprob = log_normal_prob(feats, means, covars) + np.log(priors)
    lpr = np.exp(logprob.T - logprob.max(axis=1))
    responsibilities = lpr/np.sum(lpr, 0)
    return responsibilities.T
            
def FisherEncode(feats, means, covars, priors, square_root=False, normalized=False, improved=False, fast=False) :
    """  
    *** Input ***
    feat : n_samples, n_dim
    means : K_cluster, n_dim
    covars : K_cluster, n_dim
    priors : K_cluster array
    """ 
    N, D = feats.shape
    ##get posterior probabilities q for each data point and each component
    posteriors = predict_prob(feats, priors, means, covars)
    
    if fast:
        print(posteriors.shape)
        ind_most_likeli = np.argmax(posteriors, axis=1)
        posteriors = np.zeros(posteriors.shape)
        posteriors[:, ind_most_likeli] = 1
    
    us = np.empty(0)
    vs = np.empty(0)
    ## this one uses the formulation given by vlfeat
    for k, covars_k in enumerate(covars) :     
        means_k_rep = means[k, :].reshape((1, D)).repeat(N, axis=0)
        covars_k_rep = covars_k.reshape((1, D)).repeat(N, axis=0)
        post_k_rep = posteriors[:, k].reshape((N, 1)).repeat(D, axis=-1)
        
        delta = (feats-means_k_rep)/np.sqrt(covars_k_rep)
        
        uk = np.sum(post_k_rep*delta, axis=0)
        uk /= (N*np.sqrt(priors[k]))
        us = np.concatenate((us, uk))
        
        vk = np.sum(post_k_rep*(delta**2-1), axis=0)
        vk /= (N*np.sqrt(2*priors[k]))
        vs = np.concatenate((vs, vk))
        
    encodedFeats = np.concatenate((us, vs))
    
    if square_root:
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))
    if normalized:
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)    
    if improved:
        encodedFeats = np.sign(encodedFeats)*np.sqrt(abs(encodedFeats))  
        encodedFeats = encodedFeats/np.linalg.norm(encodedFeats)
    return np.asmatrix(encodedFeats)


### ----------------------------  Load Data & model --------------------------------
# Laod data for FV
with open(LLD_ROOT+'lld_'+TARGET+'.pickle', 'rb') as handle:
    Data = pickle.load(handle)

# Laod train label
with open(LAB_ROOT+'label_'+TARGET+'.pickle', 'rb') as handle:
    Label = pickle.load(handle)


### ----------------------------  Fisher Encodig --------------------------------
for MIXURE in MIXURES:
    # Load GMMM model
    with open(GMM_ROOT+'GMM_'+GMM_TARGET+'_m'+str(MIXURE)+'.pickle', 'rb') as handle:
        GMMmodel = pickle.load(handle)

    FV_data = pd.DataFrame()
    for index, (key, value) in enumerate(sorted(Data.items())):
        if key.split('_')[0] != 'test':
            fv = FisherEncode(value, GMMmodel['mean'], GMMmodel['sigma'], GMMmodel['prior'], improved = False, fast=False)
            fv_ = pd.DataFrame(np.concatenate([fv, np.asmatrix(Label[key])], axis=1), index=[key])
            FV_data = pd.concat([FV_data,fv_], axis=0)
            print index, key, Label[key]
    FV_data.to_pickle(GMM_ROOT+'FV_'+GMM_TARGET+'_'+TARGET+'_m'+str(MIXURE)+'.pickle')  # where to save it, usually as a .pkl

    
    

