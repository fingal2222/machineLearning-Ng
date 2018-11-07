# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:09:07 2018

@author: zhaof
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

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

def nnCostFunction(X,y,num_labels,Theta1,Theta2,lam):
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
        
    return J/m+lam*(np.sum(Theta1[:,1:]*Theta1[:,1:])+np.sum(Theta2[:,1:]*Theta2[:,1:]))/(2*m)

def sigmoidGradient(z):
    z=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz(1-gz) 

def randInitializeWeights(L_in,L_out):
    epsilon_init=np.sqrt(6)/np.sqrt(L_in+L_out)
    W=rd.randint(L_in,L_out)*2*epsilon_init-epsilon_init
    return W

def backPropagation(X,y,num_labels,Theta1,Theta2):
    m=np.size(X,0)
    Delt2=np.zeros([10,25])
    Delt1=np.zeros([25,400])
    for i in range(m):
        a1=X[i,:].reshape(1,len(X[i,:]))
        x_temp=np.hstack((np.ones([1,1]),a1))
        y_temp=onehot(y[i,:],num_labels)
        a2=sigmoid(np.dot(x_temp,Theta1.T))
        a2_temp=np.hstack((np.ones([np.size(a2,0),1]),a2))
        a3=sigmoid(np.dot(a2_temp,Theta2.T))
        delt3=a3-y_temp        
        delt2=np.dot(Theta2.T,delt3.T)*((a2_temp*(1-a2_temp)).T)    #26*1
       
        Delt2=Delt2+np.dot(delt3.T,a2)#10*25
        Delt1=Delt1+np.dot(delt2[1:,:],a1)#25*400
    D1=Delt1/m
    D2=Delt2/m
    return D1,D2
        

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
#    res=nnCostFunction(X,y,10,Theta1,Theta2,1)
    Theta1=np.tile(randInitializeWeights(25,401),(25,401))
    Theta2=np.tile(randInitializeWeights(10,26),(10,26))
    backPropagation(X,y,10,Theta1,Theta2)
    
    
    
    