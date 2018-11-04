# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:39:12 2018

@author: zhaof
"""
import computeCost
import numpy as np
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m=len(y)
    J_history=np.zeros([num_iters,1])
    for iter in range(num_iters):
        theta=theta-alpha/m*np.dot(X.T,np.dot(X,theta)-y.reshape(m,1))
        J_history[iter]=computeCost.computeCost(X, y, theta)
    return theta,J_history
        