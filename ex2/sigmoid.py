# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:06:19 2018

@author: zhaof
"""
import numpy as np
def sigmoid(z):
    gz=np.zeros(np.shape(z))
    gz=1/(1+np.exp(-z))
    return gz