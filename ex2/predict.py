# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:01:53 2018

@author: zhaof
"""
import numpy as np
import sigmoid as sg

def predict(X,theta):
    [m,n]=np.shape(X)
    p=np.zeros([1,m])
    loc=np.where(sg.sigmoid(np.matrix(theta)*X.T)>=0.5)
    p[loc]=1
    return p