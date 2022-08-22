# Theme-Evolution
项目--科学发展主题演化研究--代码

整个流程:   
1 过滤没有 关键词 & 摘要 & 标题 & 出版年月 的数据, 然后再将数据按照出版年月排序, 并且, 对于同一出版年月的数据, 按照DOI进行二次排序  
  然后, 将每个期刊的数据, 进行按月平均, 如: SCIENTOMETRICS期刊缺失2018-01 2018-02的数据, 则利用2018-03的数据, 平均到这三个月  
2 对每条数据的 关键词(关键词+摘要中与关键词相似度最高的3个词) 进行 去停用词 去科学常用词 词性还原 处理  
  对 摘要(标题+摘要) 进行 去停用词 去前100高频词 词性还原 保留关键词(将关键词使用 _ 连接) 处理,   
  并使用 句子(用于word2vec) & 摘要(用于node2vec & jaccard) 两种模式对 摘要 进行存储  
3 使用分段线性法对于以下折线图进行分段 -- x为出版年月, y为该出版年月的关键词种类数 -- , 确定时间分区  
  分段线性法-自顶向下TD算法：https://www.cnblogs.com/by1990/archive/2011/01/15/1936296.html  
  我基于此提了一点优化：限制每个时间分区最小必须跨越5个月份  
4 将 关键词 & 摘要(两种模式) 按照时间分区聚集  
5 对按时间分区聚集的 关键词 配合摘要 作 图网络 & 模型  
  word2vec: 利用每个时间分区的 摘要(句子集合) 做训练集, 分别训练出每个时间分区的word2vec模型,   
            &emsp;模型参数: 剔除词频<=3, 二次采样阈值=1e-4,  采样窗口=3, 负采样噪声k=5, 模型类型: skip gram跳字模型, 词向量维度=100  
            &emsp;参考链接: https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/docs/chapter10_natural-language-processing/10.3_word2vec-pytorch.md  
            &emsp;利用该word2vec模型, 计算时间分区内的关键词相似度, 将所有 相似度 > 阈值0.3 的两个关键词构建边, 制作出 关键词图网络,   
            &emsp;同时, 利用相邻时间分区的 摘要(句子集合) 训练出融合年代的word2vec模型和关键词图网络  
            &emsp;利用所有 摘要(句子集合) 训练出一个全时区word2vec模型 和 全时区关键词图网络  
  node2vec: 利用每个时间分区的每个 摘要, 设置共现窗口为3, 将所有摘要词作为点, 根据是否在共现窗口出现构造边, 制作出 摘要图  
            &emsp;然后从摘要图中, 制作出仅含以关键词为节点的子图.  
               &emsp;子图的边构造如下: 如果任意两个关键词在原图中是直连的, 则子图中这两个关键词也是直连的,   
            &emsp;如果任意两个关键词在原图中不是直连的, 则判断这两个关键词的最短路中, 是否包含其他该子图的点, 如果包含, 则不处理, 如果不包含,   
            &emsp;则对这两个关键词构建新边.  
               &emsp;然后将每个时间分区内的所有子图各自合并, 形成各自时间分区的 关键词图网络. 然后将各自的 关键词图网络 做训练集,   
            &emsp;分别训练出各自时间分区的node2vec模型--node2vec模型是利用多次随机游走重新制作新句子集, 然后根据新句子集进行word2vec训练.  
            &emsp;同时, 利用相邻时间分区的 关键词图网络 训练出融合年代的node2vec模型和关键词图网络  
            &emsp;利用所有 关键词图网络 训练出一个全时区node2vec模型 和全时区关键词图网络   
            &emsp;模型参数: 负采样词数量=30  
            &emsp;参考链接: https://www.bilibili.com/video/BV1BS4y1E7tf  
  jaccard: 利用每个时间分区的每个 摘要, 设置共现窗口为3, 将所有摘要词作为点, 根据是否在共现窗口出现进行构造边, 制作出 摘要图  
           &emsp;然后从摘要图中, 制作出仅含以关键词为节点的子图.  
               &emsp;子图的边构造如下: 如果任意两个关键词在原图中是直连的, 则子图中这两个关键词也是直连的,   
           &emsp;如果任意两个关键词在原图中不是直连的, 则判断这两个关键词的最短路中, 是否包含其他该子图的点, 如果包含, 则不处理, 如果不包含,   
           &emsp;则对这两个关键词构建新边.  
               &emsp;然后将每个时间分区内的所有子图各自合并, 形成各自时间分区的 关键词图网络. 同时,   
           &emsp;利用相邻时间分区的 关键词图网络 制作融合年代的关键词图网络, 利用所有 关键词图网络 制作全时区关键词图网络.  
           &emsp;并且, jaccard不制作模型.  
6 对每个时间分区的 关键词图网络 进行二点连通-louvain划分, 得到每个时间分区的社区子图,   
    &emsp;使用度中心性(代表社交关联强度)>0.1或z-得分(代表社交相似强度)>2.5筛选出主节点, 作为该子社区的关键词(社区代表词)  
      &emsp;定义一套新的相邻年代的演变方式: 对于相邻年代-(年代1, 年代2), 使用融合年代(两者图交集)来观察演变  
      &emsp;融合年代解释: 融合年代是两个图的并集, 它代表着新老社区的混合, 同时也隐含着新老社区的关联  
      &emsp;通过观察老年代和融合年代的社区划分, 可以知道老年代在演变过程中, 逐渐和那些词有了新的交集  
      &emsp;再通过观察新社区的社区划分, 了解到新年代的词又如何进行关联  
  &emsp;过程: 老年代社区划分, 融合年代社区划分, 新年代社区划分, 老年代和新年代进行社区代表词筛选, 融合年代不进行筛选  
    &emsp;计算老年代每个社区与融合年代的所有社区的交集点数量, 选取 最大数量的融合年代社区 作为老年代与融合年代的演变,  
    &emsp;对于新年代也进行类似操作, 选取 最大数据的融合年代社区 作为融合年代与新年代的演变.  
    &emsp;此时, 老年代的社区和新年代的社区, 通过融合年代进行关联  
7 利用 融合年代模型, 计算相邻时间分区的 社区代表词 之间的相关性  
  &emsp;其中, word2vec & node2vec 是利用模型, 将词转化成向量, 计算两个向量之间的余弦相似度  
    &emsp;其实只要新老年代的社区通过一个融合年代进行关联, 就说明他们的词是相关的, 此时只需要计算: 对于新年代社区的词, 老年代社区哪个词贡献最大即可  
  &emsp;jaccard是利用 融合年代图网络, 计算两个词之间的jaccard值:   
      &emsp;如果这两个词在图网络中相邻, 则jaccard值 = len(词1邻居∩词2邻居) / len(词1邻居∪词2邻居)  
      &emsp;如果这两个词在图网络中最短路距离为2, 则jaccard值= 0.5 * len(词1邻居∩词2邻居) / len(词1邻居∪词2邻居)  
      &emsp;如果这两个词在图网络中最短路距离>2, 则jaccard值= 0  
      &emsp;我们依旧选取 值最大 作为新老词的关联  
8 利用6的演变, 制作时间社区演绎图: 该图只包含有边相连的社区点, 主要用来发现 社区的演绎  
9 利用6的演变, 制作时间社区全状态演绎图: 该图包含所有的 社区 , 同时根据颜色, 显示出社区演绎的六种状态  
  &emsp;新生: 红色   消亡: 蓝色   继承: 绿色   分裂: 紫色   融合: 黄色   孤立: 棕色   其他: 灰色   融合中介社区: 粉色  
10 统计了每个时区的关键词词频情况, 可以直观感受关键词的演绎, 主要利用该图评价我们算法的演绎情况  
