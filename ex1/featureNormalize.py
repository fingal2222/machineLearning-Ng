# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:06:00 2018

@author: zhaof
"""
import numpy as np
from numpy.matlib import repmat

def featureNormalize(X):
    X_norm=X
    mu=np.zeros([1,np.size(X,1)])
    sigma=np.zeros([1,np.size(X,1)])
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-np.tile(mu,(np.size(X,0),1)))/np.tile(sigma,(np.size(X,0),1))
    return X_norm,mu,sigma
    
    
    