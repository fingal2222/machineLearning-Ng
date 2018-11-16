# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:36:32 2018

@author: zhaof
"""
import re

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
    
    #tokensize email
    
    if email_contents.strip()!='':
        
    
    