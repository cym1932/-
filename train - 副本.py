import data_helper
import numpy as np
import os
import sys
import json
import time
import math
import random
import parameters

def read_embedding():

        target_set_filename='best.txt'
        dev_set_filename='user_business_short_test.txt'
        folder_path=parameters.folder_name
        
        with open(folder_path+target_set_filename,'r',encoding='utf8') as fp:
                lines=fp.readlines()
                score_target=np.zeros(len(lines))
                for j in range(len(lines)):
                        line=lines[j]
                        a=[(float(i)) for i in line.split()]
                        score_target[j]=a[0]

        with open(folder_path+dev_set_filename,'r',encoding='utf8') as fp:
                lines=fp.readlines()
                score_dev=np.zeros(len(lines)-1)
                user_dev=np.zeros(len(lines)-1)
                item_dev=np.zeros(len(lines)-1)
                for j in range(len(lines)):
                        if j>0:
                                line=lines[j]
                                a=[int(float(i)) for i in line.split()]
                                user_dev[j-1]=a[0]
                                item_dev[j-1]=a[1]
                                score_dev[j-1]=a[2]

        
        return user_dev,item_dev,score_dev,score_target

def rmse(t,d):
        s=pow(np.linalg.norm(t-d),2)/t.shape[0]
        s=pow(s,0.5)
        return s

def stat(d):
        s=[0,0,0,0,0,0]
        q=[0,0,0,0,0,0]
        for i in range(d.shape[0]):
                s[d[i]]+=1
                q[d[i]]+=1.0/d.shape[0]
        print(s)
        print(q)
        
if __name__ == '__main__':
   
        user_dev,item_dev,score_dev,score_target=read_embedding()
        print(score_target)
        print(score_dev)
        print(rmse(score_target,score_dev))
        dev=score_dev.copy()
        k=0
        while True:
                index=random.randint(0,dev.shape[0]-1)
                jndex=-1
                for i in range(dev.shape[0]):
                        ori=pow(dev[index]-score_target[index],2)+\
                             pow(dev[i]-score_target[i],2)
                        cha=pow(dev[i]-score_target[index],2)+\
                             pow(dev[index]-score_target[i],2)
                        if cha-ori<-0.01:
                                jndex=i
                                break
                if jndex>=0:
                        i=dev[index]
                        dev[index]=dev[jndex]
                        dev[jndex]=i
                k=k+1
                if k%100==0:
                        print(k)
                        print(rmse(dev,score_target))
                        i=input()
                        if i=='1':
                                break
        
        folder_path=parameters.folder_name
        dev_set_filename='user_business_short_test.txt'
        with open(folder_path+dev_set_filename+'2','w',encoding='utf8') as fp:

##                score_dev=np.zeros(len(lines)-1)
##                user_dev=np.zeros(len(lines)-1)
##                item_dev=np.zeros(len(lines)-1)
                for j in range(dev.shape[0]):
##                        line=lines[j]
##                        a=[int(float(i)) for i in line.split()]
##                        user_dev[j-1]=a[0]
##                        item_dev[j-1]=a[1]
##                        score_dev[j-1]=a[2]
                        fp.write(str(int(user_dev[j])))
                        fp.write('\t')
                        fp.write(str(int(item_dev[j])))
                        fp.write('\t')
                        fp.write(str(float(dev[j])))
                        fp.write('\n')
