# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:08:22 2018

@author: zhaof
"""
# sum(-ylogh-(1-y)log(1-h))/m
import numpy as np
import sigmoid as sg
def costFunction(theta,X,y):
    m=len(y)
    z=np.dot(X,theta)
    h=sg.sigmoid(z)
    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
#    y=y.reshape(m,1)
#    print(np.shape(y))
#    J=0
#    
#    grad=np.zeros(np.shape(theta))  
#    J=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))
    grad=np.dot(X.T,(h-y))/m
    return J,grad
    
    