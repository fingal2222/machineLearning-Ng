# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:40:01 2018

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

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=np.zeros([num_iters,1])
    theta_s=theta
    y_shape=np.shape(y)
    for iter in range(num_iters):
        theta[0]=theta[0]-alpha/m*np.sum(np.dot(X,theta_s).reshape(y_shape)-y)        
        theta[1]=theta[1]-alpha/m*np.sum(np.multiply(np.dot(X,theta_s).reshape(y_shape)-y,X[:,1].reshape(y_shape)))
        theta_s=theta
        J_history[iter]=computeCost(X,y,theta)
        
    return theta,J_history
    

b=np.loadtxt("D:\BaiduNetdiskDownload\homework\machineLearning-Ng\ex1\ex1data1.txt",delimiter=",")
X=b[:,0]
y=b[:,1]
m=len(y) # number of training examples

##### Plot Data
plt.scatter(b[:,0],b[:,1])
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.title('')
plt.show()


###Part 3: Gradient descent
[m,n]=np.shape(b[:,0:1])
X=np.append(np.ones([m,1]),b[:,0]).reshape(n+1,m).T #Add a column of ones to x ???
theta=np.zeros([2,1])  #initialize fitting parameters
###Some gradient descent settings
iterations=1500;
alpha=0.01;

###compute and display initial cost
J=computeCost(X, y, theta)

[theta,J] = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ');
print('%f %f \n', theta[0], theta[1]);
###Plot the linear fit
plt.plot(X[:,1].reshape(m,1), np.dot(X,theta))
plt.scatter(b[:,0],b[:,1])
plt.show()

predict1 = np.dot(np.array([1, 3.5]),theta)
predict2 = np.dot(np.array([1, 7]),theta)
print('For population = 35,000, we predict a profit of %f\n',predict1*10000);
print('For population = 70,000, we predict a profit of %f\n',predict2*10000);

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t=np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i,j]=computeCost(X,y,t)


J_vals=J_vals.T

fig1=plt.figure()#创建一个绘图对象
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)

#至此X,Y分别表示了取样点的横纵坐标的可能取值
#用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点


plt.title("This is main title")#总标题
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('z label', color='b')#给三个坐标轴注明
plt.show()#显示模块中的所有绘图对象

plt.contourf(theta0_vals, theta1_vals, J_vals,100,alpha=0.75,cmap=plt.cm.hot)
plt.plot(theta[0],theta[1])






