# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:15:29 2018

@author: zhaof
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:13:22 2018

@author: zhaof
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def grad(theta,X,y):
    m=len(y)
    y=y.reshape(m,1)
    print(np.shape(y))
    #J=0
    grad=np.zeros(np.shape(theta))
    #J=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))
    grad=np.dot(X.T,sigmoid(np.dot(X,theta))-y)/m
    return grad

def predict(X,theta):
    [m,n]=np.shape(X)
    p=np.zeros([1,m])
    loc=np.where(sigmoid(np.matrix(theta)*X.T)>=0.5)
    p[loc]=1
    return p

def sigmoid(z):
    gz=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz

def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    # plt.scatter(np.linspace(1,m,m),b[:,0])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X[pos, 0], X[pos, 1], color='red', linewidth=2.5, marker='o', label='Admitted')
    ax.scatter(X[neg, 0], X[neg, 1], color='blue', linewidth=2.5, marker='x', label='Not Admitted')

def costFunction(theta,X,y):
    m=len(y)
    z=np.dot(X,theta)
    h=sigmoid(z)
    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
#    y=y.reshape(m,1)
#    print(np.shape(y))
#    J=0
#
#    grad=np.zeros(np.shape(theta))
#    J=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))
    grad=np.dot(X.T,(h-y))/m
    return J,grad

if __name__=='__main__':

    b=np.loadtxt("ex2data1.txt",delimiter=",")
    X=b[:,0:2]
    y=b[:,2]
    m=len(y)
    #==================== Part 1: Plotting ====================
    plotData(X,y) #可以考虑用pandas loc

    #============ Part 2: Compute Cost and Gradient ============
    [m,n]=np.shape(X)

    X=np.append(np.ones([m,1]),X,axis=1)

    initial_theta=np.zeros([n+1,1])
    theta=initial_theta
    [cost,grad]= costFunction(initial_theta,X,y)
    #[cost,grad]=costFunction.costFunction(initial_theta,X,y)
    #grad=grad.grad(initial_theta,X,y)
    #print('Cost at initial theta (zeros): %f\n' %(cost))

    #============= Part 3: Optimizing using fminunc  =============

    #cost=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))

    result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))

    #============== Part 4: Predict and Accuracies ==============
    theta=result[0]
    x=np.array([1,45,85])
    prob = sigmoid(np.sum(np.multiply(theta,x)))

    print("For a student with scores 45 and 85 , we predict an admission probability of %f" %(prob))

    p=predict(X,theta)
    accuracy=np.mean(p-np.matrix(y))
