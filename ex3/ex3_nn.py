# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 19:59:35 2018

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
    

if __name__=='__main__':
    filePathData="D:\BaiduNetdiskDownload\mlclass-ex3-jin\ex3data1.mat"
    data=loadData(filePathData)
    X=data["X"]
    y=data["y"]
    m=np.size(X,0)
    fileTheta="D:\BaiduNetdiskDownload\mlclass-ex3-jin\ex3weights.mat"
    Theta=loadData(fileTheta)
    
    Theta1=Theta["Theta1"]
    Theta2=Theta["Theta2"]
    X_temp=np.hstack((np.ones([m,1]),X))
    z2=np.dot(X_temp,Theta1.T)
    a2=sigmoid(z2)
    a2_temp=np.hstack((np.ones([np.size(a2,0),1]),a2))
    z3=np.dot(a2_temp,Theta2.T)
    a3=sigmoid(z3)
    arr=pd.DataFrame(a3)
    arr["max_index"]=np.argmax(a3,axis=1)+1
    arr["y"]=data["y"]
    eq=(arr["y"]==arr["max_index"])+0    
    prob=np.mean(eq)
    
    