# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:16:45 2018

@author: zhaof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.optimize as opt 
import sigmoid as sg

def readData(filepath):
    return pd.read_csv(filepath,sep=',',header=None)

def plotData(data):
    pos=data.loc[data[2]==1]
    neg=data.loc[data[2]==0] 
    plt.figure()
    ax1=pos.plot(kind="scatter",x=0,y=1,s=50,color="red",marker="+",label="Admitted")    
    ax2=neg.plot(kind="scatter",x=0,y=1,s=50,color="blue",marker="x",ax=ax1,label="Not Admitted")  
    ax1.set_xlabel("Microchip Test 1")
    ax2.set_ylabel("Microchip Test 2")    
#    plt.show()

def mapFeature(x1,x2):
    degree=7
    cnt=0   
    out=np.ones([28,np.size(x1)])
    for i in range(degree):
        for j in range(i+1):           
            out[cnt,:]=np.power(x1,i-j)*np.power(x2,j)
            cnt=cnt+1
    return out.T

def costFunctionReg(theta,X,y,lam):
    size=np.size(theta)
    theta=theta.reshape((size,1))    
    m=len(y)
    z=np.dot(X,theta)
    zero=np.zeros([1,1])
    theta1=np.vstack((zero,theta[1:,:]))
    h=sg.sigmoid(z) 
    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m+lam*np.dot(theta1.T,theta1)/(2*m)
    grad=np.dot(X.T,(h-y))/m+lam*theta/m
    return J,grad

def plotDecisionBoundary(theta,X,y,data):
    #分两种情况
#    a0+a1*x1+a2*x2
#    a1+a1*x1+a2*x2+a3*x1x2+.....
    plotData(data)
    paraNum=np.shape(X)[1]
    if paraNum<=3:
        #a0+a1*x1+a2*x2
        plot_x=np.array([np.min(X[:,0]),np.max(X[:,0])])
        plot_y=(-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x,plot_y)
    else:
        u=np.linspace(-1,1.5,50)
        v=np.linspace(-1,1.5,50)
        z=np.zeros(np.shape(u))
        z=np.zeros([len(u),len(v)])
#        z=np.dot(mapFeature(u,v),theta)
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j]=np.dot(mapFeature(u[i],v[j]),theta)
        plt.contour(u,v,z.T,0)
        plt.show()

if __name__=='__main__':
    filepath =('D:/BaiduNetdiskDownload/ex2-logistic regression/ex2data2.txt')
    data =readData(filepath)
#    plotData(data)
    X=data.values[:,0:2]
    x1=X[:,0]
    x2=X[:,1]     
    out=mapFeature(x1,x2)
    [row,col]=np.shape(out)
    initial_theta=np.zeros([col,1])
    lambs=np.array([0,1,100,50])
    for i in range(np.size(lambs)):
        lamb=lambs[i]   
        y=data.values[:,2:3]
        J,grad=costFunctionReg(initial_theta,out,y,lamb)    
        result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(out, y,lamb))
        plotDecisionBoundary(result[0],out,y,data)
    
    
    