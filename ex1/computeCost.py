# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:09:37 2018

@author: zhaof
"""
import numpy as np

def computeCost(X, y, theta):
    m=len(y)
    y_shape=np.shape(y)
    J=np.sum((np.dot(X,theta).reshape(y_shape)-y)**2)/(2*m)
    return J
    
    
        