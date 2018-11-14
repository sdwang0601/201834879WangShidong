# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:21:24 2018

@author: 木石之心
"""

import os
import collections
import math

''' 一些常用的路径 '''
traindata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\traindata' #划分后训练数据路径
testdata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\testdata'   #划分后测试数据路径

#返回某路径下的文件个数
def getnumfiles(datapath):
    numfiles = 0 #总文档个数 
    rootpathlist = os.listdir(datapath)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i 
        subspathlist = os.listdir(rootpath) #子文件夹列表
        numfiles = numfiles + len(subspathlist)  #计算总文档个数
    return numfiles

#返回训练集向量集合
def gettrainlist(): 
    
    #list格式为：[[str,int,float,{}],  [],[]]，
    #其中，str为某个类别，int为该类别所有文档的单词总数（一个单词在一个文档中只计算一次），
    #float为P(Ci)的值（该类别的文档总数/数据集文档总数）,
    #dict为该类别的字典，字典的value值为该类别所有单词的df值（包含该单词的文档总数）。
    rootlist = [] 
    numfiles = getnumfiles(traindata) #总文档个数 
    
    rootpathlist = os.listdir(traindata)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        
        dfdict = {} #存放值为df的类别字典
        subslist = [] #每一个类别的向量list,格式为[str,int,float,{}]
        #存类名
        subslist.append(i)  
        #存各文档的单词总数（每个单词在一个文档中只计算一次）
        wordsum = 0
        for j in subspathlist:
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词
            coun = collections.Counter(lines) #counter函数
            wordsum = wordsum + len(coun)
            for key,value in coun.items():  # counter函数的items()转化成(元素，计数值)组成的列表
                key = key.strip('\n')   #去除换行符，此时的数据是带着换行符的
                dfdict[key] = dfdict.get(key,0) + 1           
        subslist.append(wordsum)
        #存P(Xi=c)
        pc = len(subspathlist)/numfiles #P(Xi=c)为该类别文档个数/总文档个数
        subslist.append(pc)
        #存字典
        subslist.append(dfdict)
        rootlist.append(subslist)
    return rootlist

#返回测试集向量集合
def gettestlist():
    
    #list格式为[[str,[]],  [],[]]，
    #其中，str为测试集该向量所属的类别，list为测试集该向量所包含的所有单词集合（无重复单词）。
    rootlist = []
    
    rootpathlist = os.listdir(testdata)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i 
        subspathlist = os.listdir(rootpath) 
        
        for j in subspathlist:
            subslist = []   #格式为[str,[]]
            #存类名
            subslist.append(i)
            templist = []  #第一次subslist和templist写到j循环的外边，所以运行了三小时。。。。时间这么长的原因，在于增加了测试集的单词
            
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词
            coun = collections.Counter(lines) #counter函数
            for key,value in coun.items():  
                key = key.strip('\n')   
                templist.append(key)
            
            #存单词list
            subslist.append(templist)
            rootlist.append(subslist)
    return rootlist

'''
NBC算法实现 
1.装载数据；2.模型应用（伯努利朴素贝叶斯模型）；3.Laplace平滑；4.模型性能评测。
'''
def NBC():
    #装载数据
    trainlist = gettrainlist() #[[str,int,float,{}],  [],[]]
    print('训练集装载完成！ 数据集大小： ',len(trainlist))
    testlist = gettestlist() #[[str,[]],  [],[]]
    print('测试集装载完成！ 数据集大小： ',len(testlist))
    
    success = 0 #记录模型预测成功的次数
    failure = 0 #记录模型预测失败的次数
    
    print('预测开始：')
    for i in range(len(testlist)):
        maxp = 0         #与训练集向量比较后，最大的概率值P
        maxclass = ' '   #最大概率值P所属类的类名
        
        for j in range(len(trainlist)):
            #对P做log处理，不影响大小关系，不然的话用乘积，超过了计算机下限，最后全是0
            p = math.log10(trainlist[j][2]) 
            #print(p)
            for key in testlist[i][1]:
                #Laplace平滑，分子加1，分母加单词总数
                tempdp = trainlist[j][3].get(key,0) + 1 
                tempfenmu = trainlist[j][1] + len(testlist[i][1])
                tempp = math.log10(tempdp / tempfenmu)
                
                p = p + tempp
            
            #让maxp,和maxclass初始值默认为第一个测试数据的值，不可以直接设为0，因为maxp的值可能小于0
            if j == 0:
                maxp = p
                maxclass = trainlist[j][0]
            elif p > maxp:
                maxp = p
                maxclass = trainlist[j][0]
                
        #print(maxclass,' == ', testlist[i][0])
        if maxclass == testlist[i][0]:
            success = success + 1
            #print('预测成功')
        else:
            failure = failure + 1
            #print('预测失败')
        
        if (i % 1000) == 0:
            print('已完成测试次数：' ,i+1 ) #打印程序运行程度
            
    #输出模型的性能
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
预测成功次数： 14168
预测失败次数： 906
预测准确率： 0.9398965105479634
'''

