import numpy as np
from scipy.io import loadmat
import  matplotlib.pyplot as plt
import  math
from numpy.linalg import *

def loadData(filePath):
    return loadmat(filePath)

def estimateGaussian(X):
    m,n=np.shape(X)
    mu=np.zeros((n,1))
    sigma2=np.zeros((n,1))
    for i in range(n):
        mu[i,:]=np.mean(X[:,i])
        sigma2[i,:]=np.var(X[:,i])
    return mu,sigma2

def multivariateGaussian(X, mu, Sigma2):
    m,k=np.shape(mu)
    c,l=np.shape(Sigma2)
    if(c==1 or l==1):
        Sigma2=np.diag(Sigma2.ravel())
    X=X-mu.T
    bb=np.sum(np.dot(np.dot(X, pinv(Sigma2)), X.T))
    p=(2*math.pi)**(-k/2)*det(Sigma2)**(-0.5)\
      *np.exp(-0.5*np.sum(np.dot(X,pinv(Sigma2))*X,axis=1))
    return p

def visualizeFit(X,mu,sigma2):
    x=np.arange(0,35.5,0.5)
    y=np.arange(0,35.5,0.5)
    X1,X2=np.meshgrid(x,y)
    Z=multivariateGaussian(np.c_[np.ravel(X1),np.ravel(X2)],mu,sigma2)
    Z=np.reshape(Z,(np.shape(X1)))
    plt.scatter(X[:,0],X[:,1])
    plt.contour(X1,X2,Z,10**(np.arange(-20,0,3,dtype=float)))
    plt.show()
def selectThreshold(yval, pval):
    bestEpsilon=0
    bestF1=0
    F1=0
    minP=np.min(pval)
    maxP=np.max(pval)
    steps=(maxP-minP)/1000
    yval=np.ravel(yval)
    for epsilon in np.arange(minP,maxP,steps):
        # 小于epsilon为异常
        predictions=(pval<epsilon)

        tp=np.sum((predictions==1)==(yval==1))
        fp=np.sum((predictions==1)==(yval==0))
        fn=np.sum((predictions==0)&(yval==1))

        p=tp/(tp+fp)
        r=tp/(tp+fn)
        F1=2*p*r/(p+r)
        if F1>bestF1:
            bestEpsilon=epsilon
            bestF1=F1
    return bestEpsilon,F1




if __name__=='__main__':
    data = loadData("/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/ex8data1.mat")
    X=data['X']
    Xval=data['Xval']
    yval = data['yval']
    # plt.scatter(X[:,0],X[:,1])
    # plt.xlabel("Latency(ms)")
    # plt.ylabel("Throughput(mb/s)")
    # plt.show()
    mu,Sigma2=estimateGaussian(X)
    p=multivariateGaussian(X,mu,Sigma2)

    # plt.contour(X[:,0],X[:,1],np.reshape(pval,(np.size(pval),1)))

    # print(pval)
    # visualizeFit(X,mu,Sigma2)

    pval=multivariateGaussian(Xval,mu,Sigma2)
    epsilon,F1= selectThreshold(yval,pval)
    print(epsilon)

    outliers=np.where(p<epsilon)
    plt.scatter(X[outliers,0],X[outliers,1])
#Multidimensional Outliers

    data2=loadData("/Users/zhaofengjun/Documents/python/machineLearning-Ng/ex8/ex8data2.mat")
    X=data2['X']
    Xval = data2['Xval']
    yval = data2['yval']
    mu,sigma2=estimateGaussian(X)
    p=multivariateGaussian(X,mu,sigma2)
    pval=multivariateGaussian(Xval,mu,sigma2)
    epsilon,F1=selectThreshold(yval,pval)
    print(epsilon)

