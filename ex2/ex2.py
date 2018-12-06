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
    y=y.reshape((m,1))
    print(np.shape(y))
    grad=np.zeros(np.shape(theta))
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
    plt.scatter(X[pos, 0], X[pos, 1], color='red', marker='o', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], color='blue', marker='x', label='Not Admitted')
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 1 Score")
    plt.legend(loc = 'best')
#    plt.show()
def costFunction(theta,X,y):
    size=np.size(theta)
    theta=theta.reshape((size,1))
    m=len(y)
    y=y.reshape((m,1))
    z=np.dot(X,theta)
    h=sigmoid(z)
    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
    grad=np.dot(X.T,(h-y))/m
    return J,grad
def plotDecisionBoundary(theta,X,y):
    plotData(X,y)
    plot_x=np.linspace(np.min(X[:,0]),np.max(X[:,0]),1000)
    plot_y=(-1./theta[2])*(theta[1]*plot_x + theta[0])
    plt.plot(plot_x,plot_y)
    plt.show()

if __name__=='__main__':

    b=np.loadtxt("ex2data1.txt",delimiter=",")
    X=b[:,0:2]
    y=b[:,2]
    m=len(y)
    #==================== Part 1: Plotting ====================
    plotData(X,y) #可以考虑用pandas loc
    plt.show()

    #============ Part 2: Compute Cost and Gradient ============
    [m,n]=np.shape(X)
    X=np.append(np.ones([m,1]),X,axis=1)
    initial_theta=np.zeros([n+1,1])
    theta=initial_theta
    cost,grad= costFunction(initial_theta,X,y)
   
    print('Cost at initial theta (zeros): %f\n' %(cost))

    #============= Part 3: Optimizing using fminunc  =============

   
    result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
    cost,grad=costFunction(result[0],X,y)
    print("optimal parameters theta,cost is:",cost)
    #============== Part 4: Predict and Accuracies ==============
    theta=result[0]
    x=np.array([1,45,85])
    prob = sigmoid(np.sum(np.multiply(theta,x)))

    print("For a student with scores 45 and 85 , we predict an admission probability of %f" %(prob))

    p=predict(X,theta)
    accuracy=np.mean(p-np.matrix(y))
    plotDecisionBoundary(theta,b[:,0:2],b[:,2])
