# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:02:05 2018

@author: zhaof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filepath =('D:/BaiduNetdiskDownload/ex2-logistic regression/ex2data1.txt')
data =pd.read_csv(filepath,sep=',',header=None)
                        #print(data)
X = data.values[:,:2]  #(100,2)
y = data.values[:,2:3] #(100,1)
                        #print(np.shape(y))
                        #In 2
                        #%% ==================== Part 1: Plotting ====================
postive_value = data.loc[data[2] == 1]
                        #print(postive_value.values[:,2:3])
negative_value = data.loc[data[2] == 0]
                        #print(len(postive_value))
                        #print(len(negative_value))
ax1 = postive_value.plot(kind='scatter',x=0,y=1,s=50,color='b',marker="+",label="Admitted") # S is line width #https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter 
ax2 = negative_value.plot(kind='scatter',x=0,y=1,s=50,color='y',ax=ax1,label="Not Admitted")
ax1.set_xlabel("Exam 1 score")
ax2.set_ylabel("Exam 2 score")
plt.show()
                        #print(ax1 == ax2)
                        #print(np.shape(X))

                # In 3
                        #============ Part 2: Compute Cost and Gradient ===========
[m,n] = np.shape(X) #(100,2)
print(m,n)
additional_coulmn = np.ones((m,1))
X = np.append(additional_coulmn,X,axis=1)
initial_theta = np.zeros((n+1), dtype=int)
print(initial_theta)

                        # In4
                        #Sigmoid and cost function
def sigmoid(z):
    g = np.zeros(np.shape(z));    
    g = 1/(1+np.exp(-z));
    return g
def costFunction(theta, X, y):
    J = 0;
                               #print(theta)
    
    receive_theta = np.array(theta)[np.newaxis] ##This command is used to create the 1D array 
                               #print(receive_theta)
    
    theta = np.transpose(receive_theta)
                               #print(np.shape(theta))       
                               #grad = np.zeros(np.shape(theta))
    
    z = np.dot(X,theta) # where z = theta*X
                               #print(z)
    
    h = sigmoid(z) #formula h(x) = g(z) whether g = 1/1+e(-z) #(100,1)
                               #print(np.shape(h))
                               #J = np.sum(((-y)*np.log(h)-(1-y)*np.log(1-h))/m); 
    
    J = np.sum(np.dot((-y.T),np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
                               #J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
                               #error = h-y
                               #print(np.shape(error))
                               #print(np.shape(X))
    
    grad =np.dot(X.T,(h-y))/m
                               #print(grad)
    
    return J,grad
            #In5

[cost, grad] = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost)

print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros): \n',grad)

print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

             # Compute and display cost and gradient with non-zero theta

test_theta = [-24, 0.2, 0.2]
            #test_theta_value = np.array([-24, 0.2, 0.2])[np.newaxis]  #This command is used to create the 1D row array 

            #test_theta = np.transpose(test_theta_value) # Transpose 
            #test_theta = test_theta_value.transpose()
[cost, grad] = costFunction(test_theta, X, y)

print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n',grad);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


        #IN6
    # ============= Part 3: Optimizing using range  =============
import scipy.optimize as opt
    #initial_theta_initialize = np.array([0, 0, 0])[np.newaxis]
    #initial_theta = np.transpose(initial_theta_initialize)
print ('Executing minimize function...\n')
    # Working models
    #result = opt.minimize(costFunction,initial_theta,args=(X,y),method='TNC',jac=True,options={'maxiter':400})
result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
    # Not working model
    #costFunction(initial_theta,X,y)
    #model = opt.minimize(fun = costFunction, x0 = initial_theta, args = (X, y), method = 'TNC',jac = costFunction)
print('Thetas found by fmin_tnc function: ', result);
print('Cost at theta found : \n', cost);
print('Expected cost (approx): 0.203\n');