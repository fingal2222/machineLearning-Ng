# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:17:01 2018

@author: zhaof
"""

import numpy as np
import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=np.zeros([num_iters,1])
    theta_s=theta
    y_shape=np.shape(y)
    for iter in range(num_iters):
        theta[0]=theta[0]-alpha/m*np.sum(np.dot(X,theta_s).reshape(y_shape)-y)        
        theta[1]=theta[1]-alpha/m*np.sum(np.multiply(np.dot(X,theta_s).reshape(y_shape)-y,X[:,1].reshape(y_shape)))
        theta_s=theta
        J_history[iter]=computeCost.computeCost(X,y,theta)
        
    return theta,J_history
    