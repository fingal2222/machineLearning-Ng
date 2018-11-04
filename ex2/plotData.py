# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:50:36 2018

@author: zhaof
"""
import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
    pos=np.where(y==1)
    neg=np.where(y==0)
    #plt.scatter(np.linspace(1,m,m),b[:,0])
    fig, ax = plt.subplots(figsize=(12,8)) 
    ax.scatter(X[pos,0],X[pos,1],color='red',linewidth=2.5,marker='o',label='Admitted')
    ax.scatter(X[neg,0],X[neg,1],color='blue',linewidth=2.5,marker='x',label='Not Admitted')
    