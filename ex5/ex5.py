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
    J=0
    theta = theta.reshape((np.size(theta), 1))   
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
    grad=np.zeros(np.shape(theta))
    z = np.dot(X, theta)
    theta1 = theta
    if lam != 0:
        zero = np.zeros([1, 1])
        theta1 = np.vstack((zero, theta[1:, :]))

    grad = np.dot(X.T, z - y) / m + lam *theta1 / m
    return  np.ravel(grad)

def trainLinearReg(X,y,lamb):
    initial_theta=np.zeros((np.size(X,1),1))
    return opt.fmin_cg(lineRegCostFunction, x0=np.ravel(initial_theta), fprime=gradient, args=(X, y, lamb))

def learningCurve(X,y,Xval,yval,lamb):
    m=np.size(X,0)
    J_train=np.zeros((m,1))
    J_val=np.zeros((m,1))

    for i in range(m):
        X_temp=X[0:i,:]
        y_temp=y[0:i,:]
        all_theta=trainLinearReg(X_temp,y_temp,lamb)
        J_train[i,:]= lineRegCostFunction(all_theta,X_temp,y_temp,0)
        J_val[i, :] = lineRegCostFunction(all_theta, Xval, yval, 0)
    return J_train,J_val


def polyFeatures(X,p):
    X_temp=np.zeros((np.size(X,0),p))
    for i in range(p):
        X_temp[:,i]=X[:,0]**(i+1)
    return X_temp

def featureNormalize(X):
    mu=np.mean(X,axis=0)
    X_norm=X-mu
    sigma=np.std(X_norm,axis=0)
    return X_norm/sigma,mu,sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x=np.linspace(min_x - 15,max_x + 25,num=(max_x-min_x+40)/0.05)
    x_poly=polyFeatures(x.reshape((len(x),1)),p)
    x_poly=x_poly-mu
    x_poly=x_poly/sigma    
    x_poly=np.hstack((np.ones((np.size(x_poly,0),1)),x_poly))
    plt.plot(x,np.dot(x_poly,theta))    
    
def validationCurve(X,y,Xval,yval,lamArray):
    m=np.size(lamArray)
    J_train=np.zeros((m,1))
    J_val=np.zeros((m,1))
    for i in range(m):
        all_theta=trainLinearReg(X,y,lamArray[i])
        J_train[i,:]= lineRegCostFunction(all_theta,X,y,0)
        J_val[i, :] = lineRegCostFunction(all_theta, Xval, yval, 0)
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

    m=len(y)
    #Plot training data
    plt.scatter(X,y)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")

    # plt.subplot(3, 1, 1)
    # plt.scatter(Xval, yval)
    # plt.subplot(3, 1, 2)
    # plt.scatter(Xtest, ytest)
    # plt.subplot(3, 1, 3)
    # plt.scatter(X, y)
    # plt.show()
    
# =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 

    theta=np.ones((2,1))   
    J=lineRegCostFunction(theta,np.hstack((np.ones((m,1)),X)),y,1)
    print("Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n ")
    print(J)
    
    
#   =========== Part 3: Regularized Linear Regression Gradient =============
#%  You should now implement the gradient for regularized linear 
#%  regression.
    theta=np.ones((2,1))   
    J=lineRegCostFunction(theta,np.hstack((np.ones((m,1)),X)),y,1)
    grad=gradient(theta,np.hstack((np.ones((m,1)),X)),y,0)
    print("Gradient at theta = [1 ; 1]:  [%f; %f]")
    print("\n(this value should be about [-15.303016; 598.250744])\n")
    print(grad[0],grad[1])
    
    #
    
# %% =========== Part 4: Train Linear Regression =============
#%  Once you have implemented the cost and gradient correctly, the
#%  trainLinearReg function will use your cost function to train 
#%  regularized linear regression.
#% 
#%  Write Up Note: The data is non-linear, so this will not give a great 
#%                 fit.
#%
#
#%  Train linear regression with lambda = 0
   
    lam=1
    alltheta=trainLinearReg(np.hstack((np.ones((m,1)),X)),y,lam)
#    %  Plot fit over the data
    plt.scatter(X,y)
    plt.plot(X,np.dot(np.hstack((np.ones((m,1)),X)),alltheta))
    plt.show()    
   
#   %% =========== Part 5: Learning Curve for Linear Regression =============
#%  Next, you should implement the learningCurve function. 
#%
#%  Write Up Note: Since the model is underfitting the data, we expect to
#%                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
#%
    lam=0.01
    error_train,error_val=learningCurve(np.hstack((np.ones((m,1)),X)),y,np.hstack((np.ones((np.size(Xval),1)),Xval)),yval,lam)
    plt.plot(range(1,m+1),error_val) 
    plt.plot(range(1,m+1),error_train)
    plt.ylim(0,150)
    plt.xlim(0,13)
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.legend('Train', 'Cross Validation')
    plt.title("Learning curve for linear regression")
    plt.show()
    for i in range(m):
        print("\t%d\t\t%f\t%f\n",i,error_train[i],error_val[i])
    
#    %% =========== Part 6: Feature Mapping for Polynomial Regression =============
#%  One solution to this is to use polynomial regression. You should now
#%  complete polyFeatures to map each example into its powers
#%
    p=8    
    # Map X onto Polynomial Features and Normalize    
    X_poly=polyFeatures(X,p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly=np.hstack((np.ones((m,1)),X_poly))
    
    #Map X_poly_test and normalize (using mu and sigma)
    X_poly_test=polyFeatures(Xtest,p)
    X_poly_test =(X_poly_test-mu)/sigma
    X_poly_test=np.hstack((np.ones((np.size(Xtest),1)),X_poly_test))
    
    #Map X_poly_val and normalize (using mu and sigma)
    X_poly_val=polyFeatures(Xval,p)
    X_poly_val= (X_poly_val-mu)/sigma
    X_poly_val=np.hstack((np.ones((np.size(Xval),1)),X_poly_val))
    
    print('Normalized Training Example 1:\n')
    print('  %f  \n', X_poly[0, :])
    
#    %% =========== Part 7: Learning Curve for Polynomial Regression =============
#%  Now, you will get to experiment with polynomial regression with multiple
#%  values of lambda. The code below runs polynomial regression with 
#%  lambda = 0. You should try running the code with different values of
#%  lambda to see how the fit and learning curve change.
#%
    
    lam=100
    theta=trainLinearReg(X_poly,y,lam)
#    % Plot training data and fit
    plt.scatter(X,y)
    plotFit(min(X), max(X), mu, sigma, theta, p)
    plt.xlim(-80,80)
    plt.ylim(-60,40)
    plt.xlabel('Change in water level (x)');
    plt.ylabel('Water flowing out of the dam (y)');
    plt.title('Polynomial Regression Fit (lambda = %f)');
    plt.show()
    
    lam=0.01
    error_train,error_val=learningCurve(np.hstack((np.ones((m,1)),X_poly)),y,np.hstack((np.ones((np.size(Xval),1)),X_poly_val)),yval,lam)
    plt.plot(range(0,m),error_val) 
    plt.plot(range(0,m),error_train)
    plt.xlim(0,12)
    plt.ylim(0,100)
    
#    %% =========== Part 8: Validation for Selecting Lambda =============
#%  You will now implement validationCurve to test various values of 
#%  lambda on a validation set. You will then use this to select the
#%  "best" lambda value.
    lamArray=np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    error_train,error_val= validationCurve(X_poly,y,X_poly_val,yval,lamArray)
    TrainCross,=plt.plot(lamArray,error_train)
    Validation,=plt.plot(lamArray,error_val)
    plt.xlim(0,10)
    plt.ylim(0,20)
    plt.legend(handles=[TrainCross,Validation],labels=["TrainCross","Validation"])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    
    
#=========== Part 9: Computing test set error and Plotting learning curves with randomly selected examples=============
#
#% Map X_poly_test and normalize (using mu and sigma)
    
    theta=trainLinearReg(X_poly,y,3)
    error_val=lineRegCostFunction(theta,X_poly_val,yval,0)    
    error_test=lineRegCostFunction(theta,X_poly_test,ytest,0)
    
    
    
    
    
    




