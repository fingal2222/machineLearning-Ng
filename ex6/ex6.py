# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:48:32 2018

@author: zhaof
"""


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from numpy import random
from sklearn import svm

def loadData(filePath):
    return loadmat(filePath)  

def plotData(X,y):
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    Xneg=X.iloc[np.where(y==0)[0]]
    Xpos=X.iloc[np.where(y==1)[0]]
    ax=Xneg.plot.scatter(x=0,y=1,color='y',marker='o')
    Xpos.plot.scatter(x=0,y=1,ax=ax,marker='x')
    

def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
# 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
# 然后画出图
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlim(X[:, 0].min(),X[:, 0].max())
    plt.ylim(X[:, 1].min(),X[:, 1].max())
    plt.scatter(X[:, 0], X[:, 1], c=np.ravel(y), cmap=plt.cm.Spectral)
    
def gaussianKernel(X1,X2,gamma):
    return np.exp(-np.sum((X1-X2)**2)/(2*gamma**2))
    
def dataset3Params(X, y, Xval, yval):
    C_vec =np.array([0.01, 0.03, 0.1, 0.3, 1, 3 ,10, 30])  
    sigma_vec =np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    error_val = np.zeros((len(C_vec),len(sigma_vec)))
#    error_train = np.zeros((len(C_vec),len(sigma_vec)))
    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            clf = svm.SVC(C=C_vec[i], kernel='rbf', gamma=sigma_vec[j], decision_function_shape='ovr')
            clf.fit(X, y.ravel())
            predict=clf.predict(Xval)
            error_val[i,j]=np.mean(yval!=predict)
            
        
    index=np.where(error_val==np.min(error_val))
    rowindex=index[0][0]
    colindex=index[1][0]
    return C_vec[rowindex],sigma_vec[colindex]
    
        
        

if __name__=='__main__':
#    filePath='ex6data1.mat'
#    data=loadData(filePath)   
#    
#    X=data['X']
#    y=data['y']  
#    plotData(X,y)
#    
#    clf = svm.LinearSVC(C=100,random_state=0)
#    clf.fit(X, y.ravel())
#    clf.score(X, y)
#    #y_pred= clf.predict(X) #预测
#    ##根据训练出的模型绘制样本点
#    plot_decision_boundary(lambda x: clf.predict(x))
#    
#    
#    filePath='ex6data2.mat'
#    data=loadData(filePath)   
#    
#    X=data['X']
#    y=data['y']  
#    plotData(X,y)
#    clf = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')
#    clf.fit(X, y.ravel())
#    plot_decision_boundary(lambda x: clf.predict(x))
    
    
    filePath='ex6data3.mat'
    data=loadData(filePath)   
    
    X=data['X']
    y=data['y'] 
    Xval=data['Xval']
    yval=data['yval']
    
    C_vec,sigma_vec=dataset3Params(X, y, Xval, yval)
    clf = svm.SVC(C=C_vec, kernel='rbf', gamma=sigma_vec, decision_function_shape='ovr')
    clf.fit(X, y.ravel())
    plot_decision_boundary(lambda x: clf.predict(x))
    
    
    
    
    
    
    
    
    
    
    