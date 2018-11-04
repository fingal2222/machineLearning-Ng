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
import plotData
import costFunction 
#import grad
import sigmoid as sg
import scipy.optimize as opt  
b=np.loadtxt("ex2data1.txt",delimiter=",")
X=b[:,0:2]
y=b[:,2]
m=len(y)

#==================== Part 1: Plotting ====================
plotData.plotData(X,y) #可以考虑用pandas loc

#============ Part 2: Compute Cost and Gradient ============
[m,n]=np.shape(X)

X=np.append(np.ones([m,1]),X,axis=1)

initial_theta=np.zeros([n+1,1])
theta=initial_theta
[cost,grad]= costFunction.costFunction(initial_theta,X,y)
#[cost,grad]=costFunction.costFunction(initial_theta,X,y)
#grad=grad.grad(initial_theta,X,y)
#print('Cost at initial theta (zeros): %f\n' %(cost))

#============= Part 3: Optimizing using fminunc  =============

#cost=-1/m*np.sum(y*np.log(sg.sigmoid(np.dot(X,theta)))+(1-y)*(np.log(1-sg.sigmoid(np.dot(X,theta)))))
   
result = opt.fmin_tnc(func=costFunction.costFunction, x0=initial_theta, args=(X, y))

#============== Part 4: Predict and Accuracies ==============
theta=result[0]
x=np.array([1,45,85])
prob = sg.sigmoid(np.sum(np.multiply(theta,x)))

print("For a student with scores 45 and 85 , we predict an admission probability of %f" %(prob))

import predict
p=predict.predict(X,theta)
accuracy=np.mean(p-np.matrix(y)) 
