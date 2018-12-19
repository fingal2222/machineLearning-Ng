# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:09:07 2018

@author: zhaof
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
 

def displayData(X):
    example_width = int(np.round(np.sqrt(np.size(X, 1))))
    [m,n]=np.shape(X)
    example_height=int(n/example_width)
    display_rows=int(np.floor(np.sqrt(m)))
    display_cols=int(np.ceil(m/display_rows))
    
    pad=1    
    display_array=-np.ones((pad + display_rows * (example_height + pad),pad + display_cols * (example_width + pad)))
    curr_ex=0
    for j in range(1,display_rows+1):
        for i in range(1,display_cols+1):
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
        plt.imshow(display_array.T)


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
def gradFunction2(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):
#    将y转成one-hot    
    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    m=np.size(X,0)

    Theta1_grad=np.zeros((np.shape(Theta1)))
    Theta2_grad=np.zeros((np.shape(Theta2)))     
    
    X=np.hstack((np.ones([m,1]),X))
    
    # 先把theta(1)拿掉，不参与正则化
    temp1=np.hstack((np.zeros([np.size(Theta1,0),1]),Theta1[:,1:]))
    temp2=np.hstack((np.zeros([np.size(Theta2,0),1]),Theta2[:,1:]))
    #计算每个参数的平方，再就求和
    temp1=np.sum(temp1**2)
    temp2=np.sum(temp2**2) 
    
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
    
    return grad  

def costFunction2(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):
#    将y转成one-hot    
    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    m=np.size(X,0)
    J=0
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
     
    cost=y_temp*np.log(a3)+(1-y_temp)*np.log((1-a3))
    
    J=-1/m*np.sum(cost)+lam/(2*m)*(temp1+temp2)
        
    return J   


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
def randDebugInitializeWeights(L_out,L_in):
    W=np.sin(range(1,L_out*(L_in+1)+1)).reshape((L_in+1,L_out)).T/10
    return W

def randInitializeWeights(L_in,L_out):
    epsilon_init=np.sqrt(6)/np.sqrt(L_in+L_out)
    W=np.random.random(size=(L_out,L_in+1))*2*epsilon_init-epsilon_init
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
        numgrad[p]=(loss2-loss1)/(2*e)
        perturb[p]=0
    return numgrad
        
def costFunction(input_layer_size,hidden_layer_size,num_labels,X,y,lam):
    def nnCostFunc(nn_params):
        return nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)[0]
    return nnCostFunc  
        
def transformVector(Theta1,Theta2):
    return np.ravel(np.concatenate((Theta1.reshape(np.size(np.matrix(Theta1)),1),Theta2.reshape(np.size(np.matrix(Theta2)),1)),axis=0))
    
        
    
def checkNNGradients(lam):
    input_layer_size=3
    hidden_layer_size=5
    num_labels=3
    m=5
    
    Theta1=randInitializeWeights(input_layer_size,hidden_layer_size)
    Theta2=randInitializeWeights(hidden_layer_size,num_labels)
#    Theta1=random.random(size=(hidden_layer_size,input_layer_size+1))*2*0
#    Theta2=random.random(size=(num_labels,hidden_layer_size+1))*2*0
    
    X=randInitializeWeights(input_layer_size-1,m)
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
    
    ##=====Loading and Visualizing Data===========
    filePathData="D:\BaiduNetdiskDownload\mlclass-ex4-jin\ex4data1.mat"
    data=loadData(filePathData)
    X=data["X"]    
    displayData(X[np.random.randint(np.shape(X)[0],size=100),:])
    
    ##=====Loading Parameters===========
    y=data["y"]
    m=np.size(X,0)
    fileTheta="D:\BaiduNetdiskDownload\mlclass-ex4-jin\ex4weights.mat"
    Theta=loadData(fileTheta)
    
    Theta1=Theta["Theta1"]
    Theta2=Theta["Theta2"]
    
    nn_params=transformVector(Theta1,Theta2) 
    
    ##=====Compute Cost===========
    lam=0
    J,grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lam)
    
    print('Cost at parameters (loaded from ex4weights):\n(this value should be about 0.287629)\n', J)

    #=========Implement Regularization
    lam=1
    J,grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lam)
    
    print('Cost at parameters (loaded from ex4weights): \n(this value should be about 0.383770', J)

    #======Sigmoid Gradient
    g=sigmoidGradient(np.array([1, -0.5,0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ',g)
    
    #====Initializing Pameters   
    
    initial_Theta1=randInitializeWeights(input_layer_size,hidden_layer_size)
    initial_Theta2=randInitializeWeights(hidden_layer_size,num_labels)    
    initial_nn_params=transformVector(initial_Theta1,initial_Theta2) 
    #=====Implement Backpropagation
    lam=0
    checkNNGradients(lam)
    #=======Implement Regularization    
    lam=3
    checkNNGradients(lam)    
    debug_J  = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, lam)
    print("\nCost at (fixed) debugging parameters (w/ lambda = 10): \n(this value should be about 0.576051)\n",debug_J[0])
   
    #Training NN   
    lam=0
   
    nn_params=opt.fmin_cg(costFunction2,x0=np.ravel(initial_nn_params),fprime=gradFunction2,args=(input_layer_size,hidden_layer_size,num_labels,X,y,lam),maxiter=50)
    
    ##nn_params,cost=TrainLinearReg(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
    
    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,(input_layer_size+1)))
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    
    #======Visualize Weights
    displayData(Theta1[:, 1:])   
    
    #======Implement Predict
    pred = predict(Theta1, Theta2, X)   
    
    
    print("\nTraining Set Accuracy: " ,np.mean((pred == y.ravel())) * 100)
    #backPropagation(X,y,10,Theta1,Theta2)
    
    
    
    