# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:44:41 2018

@author: zhaof
"""

"""
python 里的index是从0开始的，而label中是用10来表示0的，注意注意
"""



from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd

def sigmoid(z):
    gz=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz   

def displayData(X):
    example_width = int(np.round(np.sqrt(np.size(X, 1))))
    [m,n]=np.shape(X)
    example_height=int(n/example_width)
    display_rows=int(np.floor(np.sqrt(m)))
    display_cols=int(np.ceil(m/display_rows))
    
    pad=1    
    display_array=-np.ones((pad + display_rows * (example_height + pad),pad + display_cols * (example_width + pad)))
    curr_ex=0
    for j in range(1,display_rows+1):
        for i in range(1,display_cols+1):
            if curr_ex>m:
                break
            max_val=np.max(abs(X[curr_ex,:]))
            colstart=pad + (j - 1)*(example_height + pad)
            colend=pad + (j - 1)*(example_height + pad)+example_height
            rowstart=pad + (i - 1)*(example_width + pad)
            rowend=pad + (i - 1)*(example_width + pad)+example_width
            display_array[colstart:colend,rowstart:rowend] = X[curr_ex, :].reshape((example_height, example_width)) / max_val
            curr_ex=curr_ex+1
        if curr_ex>m:
            break
        plt.imshow(display_array.T)



def loadData(filePath):
    return loadmat(filePath)    
        
def lrCostFunction(theta,X,y,lam):
    m=len(y)
    theta=theta.reshape((np.size(theta),1))
    z=np.dot(X,theta)
    h=sigmoid(z)
    theta1=theta
    if lam!=0:
        zero=np.zeros([1,1])
        theta1=np.vstack((zero,theta[1:,:]))        
    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m+lam*np.dot(theta1.T,theta1)/(2*m)
    return J

def gradient(theta,X,y,lam):
    m=len(y)
    theta=theta.reshape((np.size(theta),1))
    z=np.dot(X,theta)
    h=sigmoid(z)
    theta1=theta
    if lam!=0:
        zero=np.zeros([1,1])
        theta1=np.vstack((zero,theta[1:,:])) 
    grad=np.dot(X.T,h-y)/m+lam*theta1/m
    return np.ravel(grad)

def oneVsAll(X, y, num_labels, lamb):
    m = np.size(X, 0)
    n = np.size(X, 1)
    all_theta=np.zeros([num_labels,n+1])
    X=np.hstack((np.ones([m,1]),X))
    inital_theta=np.zeros([n+1,1])
    for c in range(num_labels):
        lab=c
#        J=lrCostFunction(inital_theta,X,(y==c)+0,lamb) 
        if c==0:
            lab=10            
#        grad=gradient(inital_theta,X,(y==c)+0,lamb)         
        all_theta[c,:] = opt.fmin_cg(lrCostFunction, x0=np.ravel(inital_theta),fprime=gradient, args=(X, (y==lab)+0,lamb))
    return all_theta


def predictOneVsAll(X,all_theta):
    m=np.size(X,0)
    X=np.hstack((np.ones([m,1]),X))
    max1=sigmoid(np.dot(X,all_theta.T))   
    return max1

if __name__=='__main__':
    data=loadData("D:\BaiduNetdiskDownload\mlclass-ex3-jin\ex3data1.mat")
    X=data["X"]
    y=data["y"]
    displayData(X[np.random.randint(np.shape(X)[0],size=100),:])
    result=oneVsAll(X,y,10,0.1)
    res2=predictOneVsAll(X,result)
    
    arr=pd.DataFrame(res2)
    arr["max_index"]=np.argmax(res2,axis=1)
    arr["max_index"]= arr["max_index"].map(lambda x: x if x!=0 else 10)
    arr["y"]=data["y"]
    eq=(arr["y"]==arr["max_index"])+0    
    prob=np.mean(eq)
    print('Training Set Accuracy: ', prob* 100)
    
   