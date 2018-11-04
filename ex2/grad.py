# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:06:21 2018

@author: zhaof
"""

import numpy as np
import sigmoid as sg
def grad(theta,X,y):
    m=len(y)
    y=y.reshape(m,1)
    print(np.shape(y))
    #J=0
    grad=np.zeros(np.shape(theta))  
    #J=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))
    grad=np.dot(X.T,sg.sigmoid(np.dot(X,theta))-y)/m
    return grad