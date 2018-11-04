# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:09:07 2018

@author: zhaof
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(z):
    gz=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz   


def loadData(filePath):
    return loadmat(filePath)    

def onehot(y,num_labels):   
    size=np.size(y)
    y=y.reshape((size,1))
    out=np.zeros([size,num_labels])    
    for i in range(size):
        val=y[i,0]
        out[i,val-1]=1
    return out

def nnCostFunction(X,y,num_labels,Theta1,Theta2):
#    将y转成one-hot
    J=0
    y_temp=np.zeros([np.size(y),num_labels])
    y_temp=onehot(y,num_labels)
    X_temp=np.hstack((np.ones([m,1]),X))
    z2=np.dot(X_temp,Theta1.T)
    a2=sigmoid(z2)
    a2_temp=np.hstack((np.ones([np.size(a2,0),1]),a2))
    z3=np.dot(a2_temp,Theta2.T)
    h=sigmoid(z3)
    for i in range(num_labels):
        yi=y_temp[:,i].reshape((1,m))
        hi=h[:,i].reshape((m,1))
        J=J+np.sum(np.dot(-yi,np.log(hi))-np.dot((1-yi),np.log(1-hi)))    
    return J/m
    
    

if __name__=='__main__':
    filePathData="D:\BaiduNetdiskDownload\mlclass-ex4-jin\ex4data1.mat"
    data=loadData(filePathData)
    X=data["X"]
    y=data["y"]
    m=np.size(X,0)
    fileTheta="D:\BaiduNetdiskDownload\mlclass-ex3-jin\ex3weights.mat"
    Theta=loadData(fileTheta)
    
    Theta1=Theta["Theta1"]
    Theta2=Theta["Theta2"]
    
#    Theta1=np.zeros([25,401])
#    Theta2=np.zeros([10,26])
    res=nnCostFunction(X,y,10,Theta1,Theta2)
    
    
    