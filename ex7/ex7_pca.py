# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:36:35 2018

@author: zhaof
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def featureNormalize(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-mu)/sigma
    return X_norm,mu,sigma

def projectData(X, U, K):
    Z=np.zeros((np.size(X,0),K))
    U_reduce = U[:, 0:K]
    Z =np.dot(X, U_reduce)
    return Z
    
def recoverData(Z, U, K):
    U_reduce = U[:, 0:K]
    return np.dot(Z, U_reduce.T)

def displayData(X):
    example_width = int(np.round(np.sqrt(np.size(X, 1))))
    [m,n]=np.shape(X)
    example_height=int(n/example_width)
    display_rows=int(np.floor(np.sqrt(m)))
    display_cols=int(np.ceil(m/display_rows))
    
    pad=1    
    display_array=-np.ones((pad + display_rows * (example_height + pad),pad + display_cols * (example_width + pad)))
    curr_ex=0
    for j in range(1,display_rows):
        for i in range(1,display_cols):
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
        plt.imshow(display_array)          
      
        
if __name__=='__main__':
    K=100
    data=loadmat("D:\BaiduNetdiskDownload\mlclass-ex7-jin\ex7faces.mat")
    X=data["X"]
#    displayData(X[0:100,:])
#    plt.scatter(X[:,0],X[:,1])
#    plt.show()
    X_norm,mu,sigma=featureNormalize(X)
    m=len(X_norm)
    Sigma=np.dot(X_norm.T,X_norm)/m
    U,S,V=np.linalg.svd(Sigma)
#    plt.scatter(X_norm[:,0],X_norm[:,1])
    Z=projectData(X_norm, U, K)
    X_rec=recoverData(Z, U, K)
#    plt.plot(X_rec[:,0],X_rec[:,1])
#    plt.scatter(X_norm[:,0],X_norm[:,1])
#    plt.show()
    displayData(X_rec[0:100,:])

    
    
    
    
    #
    
    
    