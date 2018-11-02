# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:28:06 2018

@author: 木石之心
"""
import re
import nltk
import os
import collections
import math
import json

''' 
一些常用的路径
'''
oripath = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\20news-18828'     #数据原路径
traindata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\traindata' #划分后训练数据路径
testdata = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\testdata'   #划分后测试数据路径

''' 
1.预处理：
将原数据划分为训练数据和测试数据，划分范围为为训练80%，测试20%
划分方法为对于每一个文件夹的多个文档，每读取八个文档放到训练数据，就读两个文档放到测试数据
其中：
1）预处理函数pretreat（）完成划分并调用writedata(path1,path2): 函数完成写数据操作；
2）写数据函数writedata(path1,path2): path1:原数据存储路径；path2:处理后的数据的输出路径
   并用nltk进行一些文本的简单处理，eg:去停词，提取词干，去掉数字等只保留字母,并把单词转换为小写字母等
   （装nltk文件包的时候一直报错，所以没有全选，只装了用到的包）
'''

#预处理函数
def pretreat(): 
    #读文件
    rootpathlist = os.listdir(oripath)
    for i in rootpathlist:
        rootpath = oripath + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        k=1;
        for j in subspathlist:
            orisubspath = oripath + os.path.sep + i + os.path.sep + j
            if(k>=9):
                if os.path.exists(testdata + os.path.sep + i)==False:
                    os.mkdir(testdata + os.path.sep + i)    #如果不检测目录有无创建，会报错找不到目录
                testsubspath = testdata + os.path.sep + i + os.path.sep + j
                
                #调用写数据函数
                writedata(orisubspath,testsubspath)
            else:
                #print ('%s %s' % (i,j))
                if os.path.exists(traindata + os.path.sep + i)==False:
                    os.mkdir(traindata + os.path.sep + i)
                trainsubspath = traindata + os.path.sep + i + os.path.sep + j
                
                writedata(orisubspath,trainsubspath)
            k = k + 1
            if(k > 10):
                k = 1
                
#写数据函数
def writedata(path1,path2): 
    #写数据
    openw = open(path2,'w')
    openr = open(path1,'r',errors = 'ignore') #指定编码类型或者默认都报编码错误，由于文件内有特殊字符，所以选择忽略
    datalines = openr.readlines()  #按行读，每一行作为一个处理单位
    openr.close()
    
    #用nltk进行去停词，提取词干，去掉数字等只保留字母,并一律按小写字母处理
    for line in datalines:
        stopwords = nltk.corpus.stopwords.words('english') 
        porter = nltk.PorterStemmer() 
        filterwords = re.compile('[^a-zA-Z]')  
        wordsum = [porter.stem(word.lower()) for word in filterwords.split(line) if len(word)>0 and word.lower() not in stopwords]
        for s in wordsum: #对于这一行的每一个单词，调用一次写操作
            openw.write('%s\n' % s) #加入换行符，一个单词占一行
    openw.close()

''' 
2.VSM：
建立字典，计算tf-idf值，生成向量vector
1）creatidf()：
   生成带着idf值得全局字典，其中为减少数据处理，通过计算词频（词出现次数）筛选掉一些词频较小得单词
2）creatvector(inpath,outpath): inpath:原数据存储路径；outpath:生成的向量的存储路径
   遍历文本，对每一个文本中的每一个单词计算tf的值，然后计算每一个单词tf-idf值，
   生成每一个文本的tf-idf字典，对tf-idf字典进行降序排列，只选取前50个单词tfidf值，生成新的tfidf字典，
   然后加上其所属的类别作为一个向量，将其作为一条数据加入到list集中，
   最后用json的方式，将list向量集存储到对应的json文件
'''

#生成value值为idf的字典
def creatidf():
    
    #计算词频（单词总的出现次数）及Idf（整个数据集中有多少个文本包含这个单词）
    #生成按照词频筛选后的字典
    worddict = {}  #每个单词总的出现次数
    dfdict = {}    #每个单词的df
    worddfdict = {} #筛选后每个单词的df
    wordidfdict = {} #value值为idf的字典

    numfiles = 0 #总文档个数
    
    rootpathlist = os.listdir(traindata)
    for i in rootpathlist:
        rootpath = traindata + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        
        numfiles = numfiles + len(subspathlist)  #计算总文档个数
        
        for j in subspathlist:
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词
            
            #计算单词出现次数
            for s in lines:
                #print(s)
                s = s.strip()
                if s in worddict:
                    worddict[s] = worddict[s] + 1
                else:
                    worddict[s] = 1
            #计算df
            coun = collections.Counter(lines) #counter函数
            for key,value in coun.items():  # counter函数的items()转化成(元素，计数值)组成的列表
                key = key.strip('\n')   
                #print(key)
                #dfdict[key] = dfdict.get(key,0) + 1       #计算原始df
                if key in dfdict:
                    dfdict[key] = dfdict[key] + 1 
                else:
                    dfdict[key] = 1
                    
         
    #对字典做一个筛选，把那些出现次数较小的单词删去，然后赋值新的df
    print('筛选前字典大小:',len(dfdict))
    for key, value in worddict.items():
        if value > 5:
            # print(dfdict[key])
            worddfdict[key] = dfdict[key]
    #print(len(worddfdict)) 
   
    #maxvalue = worddict[max(worddict,key = worddict.get)]  #该文出现次数最多的词出现的次数
    
    #计算idf逆文档频率 = log(语料库的文档总数/(包含该词的文档数 + 1)),+1是为了防止分母为0，即所有文档都不包含该词
    for key,value in worddfdict.items():
        #wordtfidfdict[key] = (worddict[key]/maxvalue) * (math.log(numfiles/(value + 1)))
        wordidfdict[key] = math.log10(numfiles/(value + 1))
    return wordidfdict

#创建训练集和测试集的模型向量的函数
def creatvector(inpath,outpath):
    wordidfdict = creatidf() #获得value值是idf值得字典
    print('筛选后字典大小:',len(wordidfdict))
    rootlist = []  #rootlist格式为：[[str,dict],[str,dict],[str,dict]]
    
    rootpathlist = os.listdir(inpath)
    for i in rootpathlist:
        rootpath = inpath + os.path.sep + i #目录名+路径切割符+文件名;i为每一个主文件夹的路径
        subspathlist = os.listdir(rootpath) #子文件夹列表
        for j in subspathlist:
            subslist = []       #格式为：[str,dict]
            subslist.append(i)  #存放类名，i是类名，j是文本文档名
            wordtfidfdict = {}  #value值为tfidf的字典
            worddict = {}       #value值为此文本文件的每一个单词的tf值
            subspath = rootpath + os.path.sep + j
            lines = open(subspath).readlines() #此时每一行都为一个单词，lines为这个文本文件中全部单词
            
            #计算tf.采用词频标准化   词频 = 某个词在文章中出现的次数/该文出现次数最多的词出现的次数
            #计算TF-IDF = 词频（TF）* 逆文档频率（IDF）
            #print(lines)
            for s in lines:
                s = s.strip('\n')# 此处如果不去换行符，idf字典里没有换行符，那么worddict将会是空的，因为下条语句不执行
                if s in wordidfdict:
                    #print(s)
                    #s = s.strip()
                    if s in worddict:
                        worddict[s] = worddict[s] + 1
                    else:
                        worddict[s] = 1
            
            #print(len(worddict))     
            index = max(worddict, key=worddict.get)
            maxvalue = worddict[index]  #该文出现次数最多的词出现的次数
            
            #生成value为tf-idf的字典
            for key in worddict:
                wordtfidfdict[key] = (worddict[key]/maxvalue) * (wordidfdict[key]) 
            #print(type(wordtfidfdict))
            #print(len(wordtfidfdict))
            
            #降序排列，每个向量里只包含前50个词,并且把得到的元组列表重置为字典形式，必须转为字典，不然读数据较麻烦
            valueslist = sorted(wordtfidfdict.items(),key = lambda item:item[1],reverse=True)
            valuesdict = {}
            m = 0
            for key,value in valueslist:
                if m >= 50:
                    break
                m = m + 1
                valuesdict[key] = value
            #print(valuesdict)       
            #valuesdict = sorted(wordtfidfdict.items(),key = lambda item:item[1],reverse=True)
            #print(type(valuesdict))
           
            subslist.append(valuesdict)     #存放value值为tf-idf的字典 [str,dict]
            rootlist.append(subslist)       #[[str,dict],[str,dict],[str,dict]]
            
    #print(rootlist)
    #将list写入json文件
    openw = open(outpath,'w')
    json.dump(rootlist,openw,ensure_ascii=False)
    openw.close()

'''
3.KNN：
设定K的值，算出每一对向量的相似程度（余弦大小 = AB/|A|*|B|），得出预测类别，比较得出预测准确率
1）qiumo(vc)：
   vc:从json文件中读出的list，格式为：[[str,dict],[str,dict],[str,dict]]
   对于每个向量求出其模的大小，存为list
2）knn():
   设定K的值，对于测试集的每一个向量，让其与训练集里的所有向量进行比较，保留下前K个与其最相似（cos值大）的向量，
   统计出前k个中出现的最多次数的类别，此类别为模型的预测类别，让其与该向量的真实类别比较，若相同，即为预测成功，
   对于全部的测试向量，计算出预测成功次数除以总测试次数，即为预测成功率。
'''

#对数据集求每一个向量的模，返回一个list
def qiumo(vc):
    vcmo = [] #数据集每一个向量的模的list
    
    #print(type(vc))  list  [[str,dict],[str,dict],[str,dict]]
    #print(type(vc[0]))  list [str,dict]
    #print(type(vc[0][0]))  str
    #print(type(vc[0][1]))  dict
    
    for i in vc:  #i为 list [str,dict]
        mosum = 0
        #print(type(i[1]))
        for key in i[1]:
            mosum = mosum + i[1][key] * i[1][key]
        mo = math.sqrt(mosum)
        vcmo.append(mo)
    return vcmo

'''
openr = open(vctrainpath,'r')
vctrain = json.load(openr)
openr.close()
print(len(vctrain))
'''

#KNN的具体实现
def knn():
    
    k = 5
    
    #从json文件中读取训练和测试模型向量
    openr = open(vctrainpath,'r')
    vctrain = json.load(openr)
    openr.close()
    #print (vctrain)
    openr = open(vctestpath,'r')
    vctest = json.load(openr)
    openr.close()
    #print (vctest)
    
    #先把每一个向量的模求出来，存为两个列表
    motrain = qiumo(vctrain)
    motest = qiumo(vctest)
    
    #预测成功与失败的次数
    success = 0
    failure = 0
    
    for i in range(len(vctest)):
        
        cos = []
        
        for j in range(len(vctrain)):
            
            temp = []
            
            vcsum = 0 #求向量的乘积
            for key in vctest[i][1]:
                if key in vctrain[j][1]:
                    vcsum = vcsum + vctest[i][1][key] * vctrain[j][1][key]
            #求余弦 = AB/|A|*|B|
            cosij = vcsum /( motrain[j] * motest[i])
            
            #
            temp.append(vctrain[j][0])
            temp.append(cosij)
            cos.append(temp)
            
        #筛选出前k个数据，同时把前k个数据，由二元组转化为字典，关键字为类名
        #降序排列
        coslist = sorted(cos,key = lambda item:item[1],reverse=True)
        
        
        cosdict = {}
        m = 0
        for key,value in coslist:
            if m >= k:
                break
            m = m + 1
            cosdict[key] = cosdict.get(key,0) + 1 
        
        #print(m)    
        #算出k个数据中出现次数最多的类
        maxclass = ' '
        maxvalue = 0
        
        for key,value in cosdict.items():
            if value > maxvalue:
                maxclass = key
                maxvalue = value
        
        #与测试集这一条数据的类别进行对比
        if maxclass == vctest[i][0]:
            success = success + 1
            print('预测成功')
        else:
            failure = failure + 1
            print('预测失败')
        
    #预测的成功率
    successp = success / (success + failure)
    
    print('该模型的性能：')
    print('总测试次数：' , (success + failure))
    print('预测成功次数：' , success)
    print('预测失败次数：' , failure)
    print('预测准确率：' , successp)
    
#函数运行语句

#模型向量的保存路径
vctrainpath = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\vctrain\\vctrain.json' 
vctestpath = 'D:\\code\\AnacondaCodes\\Homework 1 VSM and KNN\\vctest\\vctest.json'
   
#创建向量    
creatvector(traindata,vctrainpath)
print('训练集的向量已创建完成！' )
creatvector(testdata,vctestpath)
print('测试集的向量已创建完成！' )

#运行knn              
knn()        

'''   
k = 5  vector = 50  tf > 5
总测试次数： 3754
预测成功次数： 3121
预测失败次数： 633
预测准确率： 0.8313798614810869

不同参数下的模型运行结果：

字典大小：82599  tf > 0
字典大小：14911  tf > 10
筛选后字典大小: 47537  tf > 1
筛选后字典大小: 21860  tf > 5

k = 5  vector = 50  tf > 5
该模型的性能：
总测试次数： 3754
预测成功次数： 3167
预测失败次数： 587
预测准确率： 0.8436334576451785

k = 3  vector = 50  tf > 1
该模型的性能：
总测试次数： 3754
预测成功次数： 3207
预测失败次数： 547
预测准确率： 0.854288758657432

该模型的性能：
k = 5  vector = 50  tf > 1
总测试次数： 3754
预测成功次数： 3179
预测失败次数： 575
预测准确率： 0.8468300479488545

k = 5  vector = 50  tf > 1

总测试次数： 3754
预测成功次数： 3134
预测失败次数： 620
预测准确率： 0.8348428343100692

k = 10  vector = 30  tf > 10
总测试次数： 3754
预测成功次数： 3035
预测失败次数： 719
预测准确率： 0.8084709643047416

k = 10  vector = 200  tf > 0
总测试次数： 3754
预测成功次数： 3092
预测失败次数： 662
预测准确率： 0.823654768247203

k = 10  vector = 50  tf > 10
总测试次数： 3754
预测成功次数： 3102
预测失败次数： 652
预测准确率： 0.8263185935002664

k = 10  vector = 50  tf > 0
总测试次数： 3754
预测成功次数： 3076
预测失败次数： 678
预测准确率： 0.8193926478423016

k = 15  vector = 50  tf > 0
总测试次数： 3754
预测成功次数： 3071
预测失败次数： 683
预测准确率： 0.8180607352157698

k = 20  vector = 50  tf > 0
总测试次数： 3754
预测成功次数： 3056
预测失败次数： 698
预测准确率： 0.8140649973361748

'''
            
    
    
    

            
    
    



    
                        
            
