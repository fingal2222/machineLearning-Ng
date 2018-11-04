# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:13:56 2018

@author: zhaof
"""
import numpy as np
def normalEqn(X, y):
    theta=np.zeros([np.size(X,1),1])
    theta=np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
    return theta