# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:09:07 2018

@author: zhaof
"""

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

def onehot(y,num_labels):   
    size=np.size(y)
    y=y.reshape((size,1))
    out=np.zeros([size,num_labels])    
    for i in range(size):
        val=y[i,0]
        out[i,val-1]=1
    return out

def predict(Theta1,Theta2,X):
    m=np.size(X,0) 
    h1=sigmoid(np.dot(np.hstack((np.ones([m,1]),X)),Theta1.T))
    h2=sigmoid(np.dot(np.hstack((np.ones([m,1]),h1)),Theta2.T))    
    return np.argmax(h2,axis=1)+1
    

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):
#    将y转成one-hot    
    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    m=np.size(X,0)
    J=0
    Theta1_grad=np.zeros((np.shape(Theta1)))
    Theta2_grad=np.zeros((np.shape(Theta2)))  
    y_temp=np.zeros([np.size(y),num_labels])
    y_temp=onehot(y,num_labels)
    
    X=np.hstack((np.ones([m,1]),X))
    z2=np.dot(X,Theta1.T)
    a2=sigmoid(z2)
    a2_temp=np.hstack((np.ones([np.size(a2,0),1]),a2))
    z3=np.dot(a2_temp,Theta2.T)
    a3=sigmoid(z3)
    
    # 先把theta(1)拿掉，不参与正则化
    temp1=np.hstack((np.zeros([np.size(Theta1,0),1]),Theta1[:,1:]))
    temp2=np.hstack((np.zeros([np.size(Theta2,0),1]),Theta2[:,1:]))
    #计算每个参数的平方，再就求和
    temp1=np.sum(temp1**2)
    temp2=np.sum(temp2**2)
    
#    J=np.sum(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m+lam*np.dot(theta1.T,theta1)/(2*m)
        
    cost=y_temp*np.log(a3)+(1-y_temp)*np.log((1-a3))
    
    J=-1/m*np.sum(cost)+lam/(2*m)*(temp1+temp2)
    
    delta_1=np.zeros(np.shape(Theta1))
    delta_2=np.zeros(np.shape(Theta2))
    
    for t in range(m):
        #step1
        a_1=np.matrix(X[t,:]).T
#        a_1=np.vstack((np.ones([1,1]),a_1))
        z_2=np.dot(Theta1,a_1)
        a_2=sigmoid(z_2)
        a_2=np.vstack((np.ones([1,1]),a_2))
        z_3=np.dot(Theta2,a_2)
        a_3=sigmoid(z_3)
        #step2
        err_3=np.zeros([num_labels,1])
        for k in range(num_labels):
            err_3[k]=a_3[k]-(y[t]==k+1)+0
        #step3
        err_2=np.dot(Theta2.T,err_3)
        err_2=np.multiply(err_2[1:],sigmoidGradient(z_2))
        #step4
        delta_2=delta_2+np.dot(err_3,a_2.T)
        delta_1=delta_1+np.dot(err_2,a_1.T)
    #step5
    Theta1_temp=np.hstack((np.zeros([np.size(Theta1,0),1]),Theta1[:,1:]))
    Theta2_temp=np.hstack((np.zeros([np.size(Theta2,0),1]),Theta2[:,1:]))
    Theta1_grad=1/m*delta_1+lam/m*Theta1_temp
    Theta2_grad=1/m*delta_2+lam/m*Theta2_temp
    
    grad=transformVector(Theta1_grad,Theta2_grad)
    
    return J,grad
          
        
    
#    for i in range(num_labels):
#        yi=y_temp[:,i].reshape((1,m))
#        hi=h[:,i].reshape((m,1))
#        J=J+np.sum(np.dot(-yi,np.log(hi))-np.dot((1-yi),np.log(1-hi)))
#        
#    return J/m+lam*(np.sum(Theta1[:,1:]*Theta1[:,1:])+np.sum(Theta2[:,1:]*Theta2[:,1:]))/(2*m)



def randInitializeWeights(L_out,L_in):
#    epsilon_init=np.sqrt(6)/np.sqrt(L_in+L_out)
#    W=random.random(size=(L_out,L_in+1))*2*epsilon_init-epsilon_init
    W=np.sin(range(1,L_out*(L_in+1)+1)).reshape((L_in+1,L_out)).T/10
    return W

def backPropagation(X,y,num_labels,Theta1,Theta2):
    m=np.size(X,0)
    Delt2=np.zeros([10,25])
    Delt1=np.zeros([25,400])
    for i in range(m):
        a1=X[i,:].reshape(1,len(X[i,:]))
        x_temp=np.hstack((np.ones([1,1]),a1))
        y_temp=onehot(y[i,:],num_labels)
        a2=sigmoid(np.dot(x_temp,Theta1.T))
        a2_temp=np.hstack((np.ones([np.size(a2,0),1]),a2))
        a3=sigmoid(np.dot(a2_temp,Theta2.T))
        delt3=a3-y_temp        
        delt2=np.dot(Theta2.T,delt3.T)*((a2_temp*(1-a2_temp)).T)    #26*1
       
        Delt2=Delt2+np.dot(delt3.T,a2)#10*25
        Delt1=Delt1+np.dot(delt2[1:,:],a1)#25*400
    D1=Delt1/m
    D2=Delt2/m
    return D1,D2
def computeNumericalGradient(J, theta):
    numgrad=np.zeros(np.shape(theta))
    perturb=np.zeros(np.shape(theta))
    
    e=1e-4
    for p in range(np.size(theta)):
        perturb[p]=e
        loss1=J(theta-perturb)
        loss2=J(theta+perturb)
        numgrad[p]=(loss2[0]-loss1[0])/(2*e)
        perturb[p]=0
    return numgrad
        
def costFunction(input_layer_size,hidden_layer_size,num_labels,X,y,lam):
    def nnCostFunction2(nn_params):
        return nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
    return nnCostFunction2
        
def transformVector(Theta1,Theta2):
    return np.ravel(np.concatenate((Theta1.reshape(np.size(np.matrix(Theta1)),1),Theta2.reshape(np.size(np.matrix(Theta2)),1)),axis=0))
    
        
    
def checkNNGradients(lam):
    input_layer_size=3
    hidden_layer_size=5
    num_labels=3
    m=5
    
    Theta1=randInitializeWeights(hidden_layer_size,input_layer_size)
    Theta2=randInitializeWeights(num_labels,hidden_layer_size)
#    Theta1=random.random(size=(hidden_layer_size,input_layer_size+1))*2*0
#    Theta2=random.random(size=(num_labels,hidden_layer_size+1))*2*0
    
    X=randInitializeWeights(m,input_layer_size-1)
    y=np.mod(range(1,m+1),num_labels).T+1
    
    nn_params=transformVector(Theta1,Theta2)    
    
    costFunc=costFunction(input_layer_size,hidden_layer_size,num_labels,X,y,lam)
    
    [cost,grad]=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
    
    numgrad=computeNumericalGradient(costFunc,nn_params)
    
    diff=np.linalg.norm(numgrad-grad,ord=2)/np.linalg.norm(numgrad+grad,ord=2)
    print("diff")
    print(diff)    
        

if __name__=='__main__':
    
    input_layer_size=400
    hidden_layer_size=25
    num_labels=10
    
    
    filePathData="D:\BaiduNetdiskDownload\mlclass-ex4-jin\ex4data1.mat"
    data=loadData(filePathData)
    X=data["X"]
    y=data["y"]
    m=np.size(X,0)
    fileTheta="D:\BaiduNetdiskDownload\mlclass-ex3-jin\ex3weights.mat"
    Theta=loadData(fileTheta)
    
    Theta1=Theta["Theta1"]
    Theta2=Theta["Theta2"]
    
    nn_params=transformVector(Theta1,Theta2) 
    lam=0
    
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lam)
    
#    print("J %f"%J)
    
    g=sigmoidGradient(np.array([1, -0.5,0, 0.5, 1]))
#    print("sigmoid is %f"%(g))
    
#    Theta1=np.zeros([25,401])
#    Theta2=np.zeros([10,26])
#    res=nnCostFunction(X,y,10,Theta1,Theta2,1)
    initial_Theta1=randInitializeWeights(hidden_layer_size,input_layer_size)
    initial_Theta2=randInitializeWeights(num_labels,hidden_layer_size)
    
    initial_nn_params=transformVector(initial_Theta1,initial_Theta2) 
    lam=0
    checkNNGradients(lam)
    
#    lam=3
#    checkNNGradients(lam)
#    debug_J  = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, lam)
##    print("debug_J is %f"%debug_J)
#    
#    
#    costFunc=costFunction(input_layer_size,hidden_layer_size,num_labels,X,y,lam)
#    
#    [nn_params,cost]=opt.fmin_cg(costFunc, x0=initial_nn_params)
#    
#    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)-1],(hidden_layer_size,(input_layer_size+1)))
#    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1),(num_labels,hidden_layer_size+1)])
#    
#    pred = predict(Theta1, Theta2, X)
    
    
    
#    backPropagation(X,y,10,Theta1,Theta2)
    
    
    
    