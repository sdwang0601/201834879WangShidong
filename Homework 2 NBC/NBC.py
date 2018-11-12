# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:21:24 2018

@author: 木石之心
"""

import os
import collections
import math

''' 
一些常用的路径
'''

traindata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\traindata' #划分后训练数据路径
testdata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\testdata'   #划分后测试数据路径


#返回文件个数，参数为某个路径
def getnumfiles(datapath):
    numfiles = 0 #总文档个数 
    rootpathlist = os.listdir(datapath)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        numfiles = numfiles + len(subspathlist)  #计算总文档个数
    return numfiles

#返回训练集
def gettrainlist(): 
    
    numfiles = getnumfiles(traindata) #总文档个数 
    rootlist = []
    rootpathlist = os.listdir(traindata)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        
        dfdict = {}
        subslist = []
        subslist.append(i)
        subslist.append(len(subspathlist))
        pc = len(subspathlist)/numfiles
        subslist.append(pc)
        
        
        for j in subspathlist:
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词
            
            coun = collections.Counter(lines) #counter函数, 想着输出一下coun得类型，看看是什么
            for key,value in coun.items():  # counter函数的items()转化成(元素，计数值)组成的列表,如果只写key，那现在key代表一个元组，不是索引
                key = key.strip('\n')   #去除换行符，此时的数据是带着换行符的
                dfdict[key] = dfdict.get(key,0) + 1         #筛选前的df

        subslist.append(dfdict)
        rootlist.append(subslist)
    
    return rootlist

#返回测试集
def gettestlist():
    rootlist = []
    rootpathlist = os.listdir(testdata)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        
        for j in subspathlist:
            subslist = [] 
            subslist.append(i)
            templist = []  #subslist和templist写错位置了，所以第一次运行了三小时。。。。
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词
            coun = collections.Counter(lines) #counter函数, 想着输出一下coun得类型，看看是什么
            for key,value in coun.items():  # counter函数的items()转化成(元素，计数值)组成的列表,如果只写key，那现在key代表一个元组，不是索引
                key = key.strip('\n')   #去除换行符，此时的数据是带着换行符的
                templist.append(key)
       
            subslist.append(templist)
            rootlist.append(subslist)
    return rootlist
    
def NBC():
    trainlist = gettrainlist() #[[str,int,float,{}],  [],[]]
    print('训练集装载完成！ 数据集大小： ',len(trainlist))
    testlist = gettestlist() #[[str,[]],  [],[]]
    print('测试集装载完成！ 数据集大小： ',len(testlist))
    
    success = 0
    failure = 0
    print('预测开始：')
    for i in range(len(testlist)):
        maxp = 0
        maxclass = ' '
        #biaoji = 0
        for j in range(len(trainlist)):
            p = math.log10(trainlist[j][2])
            #print(p)
            for key in testlist[i][1]:
            
                tempdp = trainlist[j][3].get(key,0) + 1
                tempfenmu = trainlist[j][1] + len(testlist[i][1])
                tempp = math.log10(tempdp / tempfenmu)
                
                p = p + tempp
                #print(p)
            if j == 0:
                maxp = p
                maxclass = trainlist[j][0]
            elif p > maxp:
                maxp = p
                maxclass = trainlist[j][0]
                
            '''
            if biaoji == 0:
                biaoji = 1
                maxp = p
                maxclass = trainlist[j][0]
            if p > maxp:
                
            '''
        #print(maxclass,' == ', testlist[i][0])
        if maxclass == testlist[i][0]:
            success = success + 1
            #print('预测成功')
        else:
            failure = failure + 1
            #print('预测失败')
        
        if (i % 1000) == 0:
            print('已完成测试次数：' ,i+1 ) #记录程序运行程度
            
    #预测的成功率
    successp = success / (success + failure)
    print('预测结束！')
    print('该模型的性能：')
    print('总测试次数：' , (success + failure))
    print('预测成功次数：' , success)
    print('预测失败次数：' , failure)
    print('预测准确率：' , successp)
                    
NBC()


'''
训练集装载完成！ 数据集大小：  20
测试集装载完成！ 数据集大小：  15074
预测开始：
已完成测试次数： 1
已完成测试次数： 1001
已完成测试次数： 2001
已完成测试次数： 3001
已完成测试次数： 4001
已完成测试次数： 5001
已完成测试次数： 6001
已完成测试次数： 7001
已完成测试次数： 8001
已完成测试次数： 9001
已完成测试次数： 10001
已完成测试次数： 11001
已完成测试次数： 12001
已完成测试次数： 13001
已完成测试次数： 14001
已完成测试次数： 15001
预测结束！
该模型的性能：
总测试次数： 15074
预测成功次数： 13779
预测失败次数： 1295
预测准确率： 0.9140904869311397
'''
