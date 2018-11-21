# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:36:32 2018

@author: zhaof
"""
import re
from nltk.stem.porter import PorterStemmer
import operator as op

def getVocabList():
    with open('vocab.txt','r',encoding='utf-8') as f:
            dic=[]
            for line in f.readlines():
                line=line.strip('\n') #去掉换行符\n
                b=line.split('\t') #将每一行以空格为分隔符转换成列表
                dic.append(b)
            print(dic)
            dic=dict(dic)
    print(dic)   
  
    
    
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
    
    #tokensize email    
    while email_contents.strip()!='':
        sStr2=' @$/#.-:&*+=[]?!(){},''">_<;%' +chr(10)+ chr(13)
        str1=email_contents[0:email_contents.find(sStr2)]
        email_contents=email_contents[email_contents.find(sStr2)+1:]
        
        #Remove any non alphanumeric characters
        rereobj = re.compile(r'[^a-zA-Z0-9]')
        str1,numbers=rereobj.subn('',str1)
        try:
            porter_stemmer = PorterStemmer()
            porter_stemmer.stem(str1)
        except:
            str1=''
            continue
        if len(str1)==1:
            continue
        #Look up the word in the dictionary and add to word_indices if found
        for in in range(len(vocabList)):
            if op.eq(vocabList[i],str1):
                word_indices=word_indices.append(i)
                
    return word_indices
            
        
            
            
    
        
        
    
    