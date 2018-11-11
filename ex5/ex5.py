from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import scipy.optimize as opt
from numpy import random

def sigmoid(z):
    gz=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def loadData(filePath):
    return loadmat(filePath)

def lineRegCostFunction(theta,X,y,lam):
    m = len(y)
    theta = theta.reshape((np.size(theta), 1))
    X=np.hstack((np.ones((m,1)),X))
    z = np.dot(X,theta)
    theta1 = theta
    if lam != 0:
        zero = np.zeros([1, 1])
        theta1 = np.vstack((zero, theta[1:, :]))
    J = np.sum((z-y)**2) / (2*m) + lam * np.dot(theta1.T, theta1) / (
                2 * m)
    return J

def gradient(theta,X,y,lam):
    m = len(y)
    theta = theta.reshape((np.size(theta), 1))
    X = np.hstack((np.ones((m, 1)), X))
    z = np.dot(X, theta)
    theta1 = theta
    if lam != 0:
        zero = np.zeros([1, 1])
        theta1 = np.vstack((zero, theta[1:, :]))

    grad = np.dot(X.T, z - y) / m + lam * np.dot(theta1.T, theta1) / m
    return  np.ravel(grad)

def trainLinearReg(X,y,initial_theta,lamb):
    return opt.fmin_cg(lineRegCostFunction, x0=np.ravel(initial_theta), fprime=gradient, args=(X, y, lamb))

def learningCurve(X,y,lamb,theta,Xval,yval):
    m=len(y)
    n=len(theta)
    all_theta=np.zeros((m,n))
    J_train=np.zeros((m,1))
    J_val=np.zeros((m,1))

    for i in range(m):
        X_temp=X[0:i,:]
        y_temp=y[0:i,:]
        alltheta[i,:]=opt.fmin_cg(lineRegCostFunction, x0=np.ravel(theta), fprime=gradient, args=(X, y, lamb))
        J_train[i,:]=lineRegCostFunction(alltheta[i,:],X_temp,y_temp,0)
        J_val[i, :] = lineRegCostFunction(alltheta[i, :], Xval, yval, 0)
    return J_train,J_val






if __name__=='__main__':
    filePath='ex5data1.mat'
    data=loadData(filePath)
    X=data['X']
    y=data['y']
    Xval=data['Xval']
    yval=data['yval']
    Xtest=data['Xtest']
    ytest=data['ytest']


    # plt.subplot(3, 1, 1)
    # plt.scatter(Xval, yval)
    # plt.subplot(3, 1, 2)
    # plt.scatter(Xtest, ytest)
    # plt.subplot(3, 1, 3)
    # plt.scatter(X, y)
    # plt.show()
    theta=np.ones((2,1))
    J=lineRegCostFunction(theta,X,y,0)
    print(J)
    grad=gradient(theta,X,y,0)
    print(grad)
    #
    alltheta=trainLinearReg(X,y,theta,0)
    print(alltheta)
    y=alltheta[0]+alltheta[1]*X
    print(y)
    # plt.plot(X,y)
    # plt.show()
    m=len(y)
    error_train,error_val=learningCurve(X,y,0,theta,Xval,yval)
    plt.plot(range(m),error_train)
    plt.plot(range(m,error_val))
    plt.show()




