项目介绍：
代码文件：Cluster.py
数据集文件：Tweets.txt
其他文件夹：

Cluster.py
代码实现的主要功能介绍如下：

1.数据集的准备
  1) 从数据集读取数据，并放到指定集合；
  2) 使用分词函数对数据集进行处理，并计算tfidf值；
  3) 输入为空，输出为tfidf数组，初始类标签集和类别数
        
2.聚类函数：调用sklearn中各个聚类函数对数据集进行处理，比较各个算法的聚类效果。
  1) 调用预处理函数，获得数据集向量，初始类标签和聚类数；
  2) 调用sklearn中的各个聚类函数；
  3) 使用NMI作为聚类效果的评价标准。

3.各个聚类算法的聚类效果对比：
  1) 使用jieba分词的聚类效果：
    kmeans算法: 0.7716101520056072
    AffinityPropagation算法: 0.777159260731288
    MeanShift算法: 0.73317315060083
    SpectralClustering算法: 0.7815326108789783
    Ward hierarchical clustering算法: 0.7811756130463107
    AgglomerativeClustering算法: 0.8229597609328941
    DBSCAN算法: 0.7658091617297429
    GaussianMixture算法: 0.7538930250947603
  2) 使用nltk分词的聚类效果：
    kmeans算法: 0.7993132547167876
    AffinityPropagation算法: 0.7836988975391973
    MeanShift算法: 0.7278222245216904
    SpectralClustering算法: 0.7771086482472876
    Ward hierarchical clustering算法: 0.7823244114906182
    AgglomerativeClustering算法: 0.8962440814686097
    DBSCAN算法: 0.737403764790242
    GaussianMixture算法: 0.7988746292609659


