# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:07:10 2018

@author: zhaof
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def computeCost(X, y, theta):
    m=len(y)
    y_shape=np.shape(y)
    J=np.sum((np.dot(X,theta).reshape(y_shape)-y)**2)/(2*m)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m=len(y)
    J_history=np.zeros([num_iters,1])
    for iter in range(num_iters):
        theta=theta-alpha/m*np.dot(X.T,np.dot(X,theta)-y.reshape(m,1))
        J_history[iter]=computeCost(X, y, theta)
    return theta,J_history

def featureNormalize(X):
    X_norm=X
    mu=np.zeros([1,np.size(X,1)])
    sigma=np.zeros([1,np.size(X,1)])
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X-np.tile(mu,(np.size(X,0),1)))/np.tile(sigma,(np.size(X,0),1))
    return X_norm,mu,sigma

def normalEqn(X, y):
    theta=np.zeros([np.size(X,1),1])
    theta=np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
    return theta

if __name__=='__main__':

    data=np.loadtxt("ex1data2.txt",delimiter=",")
    X=data[:,0:2]
    y=data[:,2]
    m=len(y) # number of training examples
    print("First 10 examples from the dataset: \n")
    #print("x = [%.0f %.0f], y = %.0f \n", %(X[0:9,:], y[0:9,:]))
    print('Program paused. Press enter to continue.\n')
    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')

    [X,mu,sigma] =featureNormalize(X)
    X=np.hstack((np.ones([m,1]),X))

    alpha=0.01
    num_iters=8500

    theta=np.zeros([3,1])

    theta,J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    plt.plot(np.linspace(1,50,50), J_history[0:50])
    plt.xlabel('Number of iterations');
    plt.ylabel('Cost J');

    #Estimate the price of a 1650 sq-ft, 3 br house
    price=np.dot(np.hstack(([1],(np.array([1650, 3])-mu)/sigma)),theta)
    print("Estimate the price of a 1650 sq-ft, 3 br house")
    print(price)

    data2=np.loadtxt("ex1data2.txt",delimiter=",")
    X=data2[:,0:2]
    y=data2[:,2]
    m=len(y)
    X=np.hstack((np.ones([m,1]),X))
    theta=normalEqn(X,y)
    price =np.dot(np.array([1,1650,3]),theta)