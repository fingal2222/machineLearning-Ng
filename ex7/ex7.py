# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:57:57 2018

@author: zhaof
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def kMeansInitCentroids(X,K):
    m=len(X)
    return X[np.random.randint(0,m,(1,K)).ravel(),:]

def findClosestCentroids(X,centroids):
    n=len(X)
    idx=np.zeros((n,1))
    for i in range(n):
        idx[i,:]=np.argmin(np.linalg.norm(X[i,:]-centroids,ord=2,axis=1))
    return idx

def computeMeans(X,idx,K):
    col=np.size(X,1)
    uk=np.zeros((K,col))
    for i in range(K):
        uk[i,:]=np.mean(X[np.where(idx==i)[0],:],axis=0).reshape(1,col)
    return uk
def plotData(X):
    data=loadmat("D:\BaiduNetdiskDownload\mlclass-ex7-jin\ex7data2.mat")
    X=data["X"]
    plt.scatter(X[0:100,0],X[0:100,1],color='y',marker='o')
    plt.scatter(X[100:200,0],X[100:200,1],marker='x')
    
    plt.scatter(X[200:300,0],X[200:300,1],color='g')
    plt.show()
    
    

def runKmeans(X,K,iterations,initial_centroids):  
    previous_centroids=np.zeros(np.shape(initial_centroids))
    for i in range(iterations):
        previous_centroids= initial_centroids       
        idx=findClosestCentroids(X,initial_centroids)
        initial_centroids=computeMeans(X,idx,K)
    for i in range(K):
        Data=X[np.where(idx==i)[0],:]
        plt.scatter(Data[:,0],Data[:,1])
    plt.show()
    return previous_centroids,idx
    
    
        

if __name__=='__main__':
    initial_centroids =np.array([[3,3],[6,2],[8,5]])
    data=loadmat("D:\BaiduNetdiskDownload\mlclass-ex7-jin\ex7data2.mat")
    X=data["X"]
#    idx=findClosestCentroids(X,initial_centroids)
#    centroids=computeMeans(X,idx,3)
    runKmeans(X,3,10,initial_centroids)
    image_path="D:\BaiduNetdiskDownload\mlclass-ex7-jin\\bird_small.png"
    img=Image.open(image_path,'r')
    img=np.array(img)
    img=img.reshape((128*128,3))
    K=16
    initial_centroids=kMeansInitCentroids(img,K)
    previous_centroids,idx=runKmeans(img,16,10,initial_centroids)
    for i in range(16):
        img[np.where(idx==i)[0],:]=previous_centroids[i]
    plt.imshow(img.reshape((128,128,3)))
        
    
    
    
      
        
    
        
            
   