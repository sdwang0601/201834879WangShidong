# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:30:16 2018

@author: 木石之心
"""
import json
import jieba 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from nltk.tokenize import word_tokenize
#注：初始打开编译环境时，如果直接在其他盘导入skleran，会显示找不到，报错，应先在c盘的py文件里运行一次

#jieba分词函数
def jieba_tokenize(text):
    #print(jieba.lcut(text))
    return jieba.lcut(text) 

''' 
数据预处理函数：
1.从数据集读取数据，并放到指定集合；
2.使用分词函数对数据集进行处理，并计算tfidf值；
3.输入为空，输出为tfidf数组，初始类标签集和类别数
'''
def pretreat(): 
    
    #Tweets数据集的读取
    openr = open('E:\\codes\\AnacondaCodes\\Homework 3\\Tweets.txt','r')
    #此处为方便读取数据集，把Tweets数据集中的每一个dict都调整为一行储存
    datalines = openr.readlines()  
    openr.close()
    textlist = []
    clusterlist = []
    k = 0
    for line in datalines:
        tempdict = json.loads(line.strip())
        textlist.append(tempdict['text'])
        clusterlist.append(tempdict['cluster'])
        if tempdict['cluster'] > k:
            k = tempdict['cluster'] 
    #print(textlist)
    #print(k) #输出结果k=110
        
    #数据集的预处理
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english',lowercase=True)
    '''  tokenizer: 指定分词函数; lowercase: 在分词之前将所有的文本转换成小写,本实验的数据集已处理为小写  '''

    tfidf_matrix = tfidf_vectorizer.fit_transform(textlist)  #需要进行聚类的文本集
    tfidf_array = tfidf_matrix.toarray() #将数据集由矩阵形式转为数组
    #print(type(tfidf_array))
    #print(tfidf_matrix)
    
    return tfidf_array, clusterlist, k

''' 
聚类函数：调用sklearn中各个聚类函数对数据集进行处理，比较各个算法的聚类效果。
1.调用预处理函数，获得数据集向量，初始类标签和聚类数；
2.调用sklearn中的各个聚类函数；
3.使用NMI作为聚类效果的评价标准。
'''
def clusterrun():
    #装载数据
    print('开始数据装载：')
    datalist, clusterlist, k = pretreat()
    print('数据装载完成！')
    print('开始执行聚类算法(使用NMI作为评价标准)：')
    
    #kmeans算法
    cluster = KMeans(n_clusters = k) # init='k-means++'，加上参数init，用k-means++方法，nmi降低了一个百分点
    # n_clusters: 指定K的值； init: 制定初始值选择的算法 
    #返回各自文本的所被分配到的类索引
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('kmeans算法:',nmi)
    #0.7749669551975551
    
    #AffinityPropagation算法
    cluster = AffinityPropagation()
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('AffinityPropagation算法:',nmi)
    #0.777159260731288
    
    #MeanShift算法
    cluster = MeanShift(bandwidth=0.5, bin_seeding=True)
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('MeanShift算法:',nmi)
    #0.73317315060083
   
    #SpectralClustering算法
    cluster = SpectralClustering(n_clusters=k,assign_labels="discretize",random_state=0)
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('SpectralClustering算法:',nmi)
    #0.7815326108789783
    
    #Ward hierarchical clustering算法
    cluster = AgglomerativeClustering(n_clusters=k, linkage='ward')
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('Ward hierarchical clustering算法:', nmi)
    #0.7811756130463107
   
    #AgglomerativeClustering算法
    cluster = AgglomerativeClustering(n_clusters=k,linkage = 'average')
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('AgglomerativeClustering算法:', nmi)
    #0.8229597609328941
     
    #DBSCAN算法
    cluster = DBSCAN(eps = 0.95, min_samples = 1 )
    clusterresult = cluster.fit_predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('DBSCAN算法:', nmi)
    #0.7658091617297429
    
    #GaussianMixture算法
    cluster = GaussianMixture(n_components=k, covariance_type='diag')
    #当type为full（默认）时，会报错，难道有的组件求不出一般协方差矩阵
    clusterresult = cluster.fit(datalist).predict(datalist)
    nmi = normalized_mutual_info_score(clusterresult, clusterlist)
    print ('GaussianMixture算法:', nmi)
    #0.7804543792671195
    print('聚类算法执行结束！')

#执行聚类函数  
clusterrun()

'''
各个聚类算法的聚类效果对比：
1.使用jieba分词的聚类效果：
开始数据装载：
数据装载完成！
开始执行聚类算法(使用NMI作为评价标准)：
kmeans算法: 0.7716101520056072
AffinityPropagation算法: 0.777159260731288
MeanShift算法: 0.73317315060083
SpectralClustering算法: 0.7815326108789783
Ward hierarchical clustering算法: 0.7811756130463107
AgglomerativeClustering算法: 0.8229597609328941
DBSCAN算法: 0.7658091617297429
GaussianMixture算法: 0.7538930250947603
聚类算法执行结束！

2.使用nltk分词的聚类效果：
开始数据装载：
数据装载完成！
开始执行聚类算法(使用NMI作为评价标准)：
kmeans算法: 0.7993132547167876
AffinityPropagation算法: 0.7836988975391973
MeanShift算法: 0.7278222245216904
SpectralClustering算法: 0.7771086482472876
Ward hierarchical clustering算法: 0.7823244114906182
AgglomerativeClustering算法: 0.8962440814686097
DBSCAN算法: 0.737403764790242
GaussianMixture算法: 0.7988746292609659
聚类算法执行结束！
'''