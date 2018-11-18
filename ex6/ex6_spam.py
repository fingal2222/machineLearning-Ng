# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:36:32 2018

@author: zhaof
"""
import re
from nltk.stem.porter import PorterStemmer
import operator as op
import numpy as np
from scipy.io import loadmat
from sklearn import svm

def getVocabList():
    with open('vocab.txt','r',encoding='utf-8') as f:
            dic=[]
            for line in f.readlines():
                line=line.strip('\n') #去掉换行符\n
                b=line.split('\t') #将每一行以空格为分隔符转换成列表
                dic.append(b)            
            dic=dict(dic)
            return dic  
  
    
    
def processEmail(email_contents):
    
    vocabList=getVocabList()
    
    email_contents=email_contents.lower()
    
    #html
    rereobj = re.compile(r'<[^<>]+>')
    email_contents,numbers=rereobj.subn(" ",email_contents)
    
    #numbers
    rereobj = re.compile(r'[0-9]+')
    email_contents,numbers=rereobj.subn("number",email_contents)
    
    #urls
    rereobj = re.compile(r'(http|https)://[^\s]*')
    email_contents,numbers=rereobj.subn('httpaddr',email_contents)
    
    #emailAddr
    rereobj = re.compile(r'[^\s]+@[^\s]+')
    email_contents,numbers=rereobj.subn("emailaddr",email_contents)
    
    #dollar
    rereobj = re.compile(r'[$]+')
    email_contents,numbers=rereobj.subn('dollar',email_contents)
    word_indices=[]
    
    
    rereobj=re.compile(r'[\@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%\r\n]+')
    email_contents,numbers=rereobj.subn(' ',email_contents)  
    li=re.split(r'[ ]+',email_contents.strip())    
                       
    #tokensize email    
    for l in range(len(li)):   
        if li[l].strip()!='':        
        #Remove any non alphanumeric characters
            rereobj = re.compile(r'[^a-zA-Z0-9]')
            str1,numbers=rereobj.subn('',li[l])
            try:
                porter_stemmer = PorterStemmer()
                porter_stemmer.stem(str1)
            except:
                str1=''
                continue
            if len(str1)==1:
                continue
            #Look up the word in the dictionary and add to word_indices if found
            for i in vocabList:
                if op.eq(vocabList[i],str1):
                    word_indices.append(i)
                
    return word_indices

def emailFeatures(word_indices,nSize):
    word_indices=np.array(word_indices,dtype=int)
    X=np.zeros((nSize,1))
    X[word_indices-1]=1
    return X
        


if __name__=='__main__':
    txt=open("D:\BaiduNetdiskDownload\homework\machineLearning-Ng\ex6\emailSample1.txt").read()
    word_indices=processEmail(txt)
    word_feature=emailFeatures(word_indices,1899)
    
    data_train=loadmat("D:\BaiduNetdiskDownload\homework\machineLearning-Ng\ex6\spamTrain.mat") 
    X=data_train["X"]
    y=data_train["y"]
    data_test=loadmat("D:\BaiduNetdiskDownload\homework\machineLearning-Ng\ex6\spamTest.mat") 
    Xtest=data_test["Xtest"]
    ytest=data_test["ytest"]
    
    
    clf = svm.LinearSVC(C=0.1,random_state=0)
    clf.fit(X, y.ravel())
    clf.score(X, y)
    p=clf.predict(X)
    nsize=len(p)
    pp=np.mean(p.reshape((nsize,1))==y)
    
    p=clf.predict(Xtest)
    nsize=len(p)
    pp=np.mean(p.reshape((nsize,1))==ytest)        
    
    #Top Predictors of Spam
    
    #Try Your Own Emails
    myEmail=open("D:\BaiduNetdiskDownload\homework\machineLearning-Ng\ex6\emailSample2.txt").read()
    word_indices=processEmail(myEmail)
    word_feature=emailFeatures(word_indices,1899)
    
    p=clf.predict(word_feature.T)
    nsize=len(p)
    pp=np.mean(p.reshape((nsize,1))==y)
    
    
    
            
        
            
            
    
        
        
    
    