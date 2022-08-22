import pickle
import numpy as np
import pandas as pd
import json
from torch_geometric.nn import Node2Vec
from itertools import combinations
import torch
import csv
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm
import networkx as nx
import re
from keybert import KeyBERT
import spacy
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, eigenvector_centrality, \
    closeness_centrality
from torch import nn
from collections import Counter
import torch.utils.data as data_use
import math
import random
import sys
import time


# 总设置
# 加载英文数据处理
nlp = spacy.load("en_core_web_lg")
kw_model = KeyBERT()


"""
工具代码, 如json_save, json_load啥
"""


def json_save(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def json_load(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data


def pickle_save(object_in, filename):
    f = open(filename, 'wb')
    pickle.dump(object_in, f)
    f.close()


def pickle_load(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f, encoding='bytes')
    f.close()
    return obj


def print_csv(file, data, num_judge=0):
    """
    :param file: 存储文件路径
    :param data: 存储数据
    :param num_judge: 0：存储数据类型为[["", ""], ["", ""]]，1：存储数据类型为["", "", ""]. 默认为0
    :return: csv数据文件
    """
    """结果列表写入文件csv"""
    with open(file, 'w', newline="") as f:
        f_csv = csv.writer(f)
        if num_judge == 0:
            f_csv.writerows(data)
        if num_judge == 1:
            for val in data:
                f_csv.writerow([val])


"""
数据预处理代码, 对应于data_preprocess()
"""


def data_concat():
    """
    :return: 将分散的数据聚合在一个csv中
    """
    data_c = []
    for i in range(11):
        data_c.append(pd.read_excel(f"./data/paper/paper{i}.xls"))
    data_c = pd.concat(data_c, axis=0, ignore_index=True)
    data_c.to_csv("./data/data_preprocess/paper.csv")


def data_filter(data_f):
    """
    :param data_f: pd数据集
    :return: 将数据集中缺失 关键词 | 摘要 | 出版年月的条目去除
    """
    data_u = data_f.copy(deep=True)
    mon_int = dict(JAN="01", FEB="02", MAR="03", APR="04", MAY="05", JUN="06",
                   JUL="07", AUG="08", SEP="09", OCT="10", NOV="11", DEC="12")
    for i in range(len(data_u)):
        data_key = data_u.loc[i, "Author Keywords"]
        data_abstract = data_u.loc[i, "Abstract"]
        data_public = data_u.loc[i, "Publication Year"]
        data_date = data_u.loc[i, "Publication Date"]
        data_source = data_u.loc[i, "Source Title"]
        if pd.isna(data_key) or pd.isna(data_abstract) or pd.isna(data_public) or pd.isna(data_date):
            data_u.drop(labels=i, axis=0, inplace=True)
        else:
            data_u.loc[i, "Source Title"] = str(data_source).upper()
            data_date = data_date.split(" ")[0]
            data_u.loc[i, "Publication Year"] = str(int(str(int(data_public))+mon_int[data_date]))
    data_u.reset_index(inplace=True, drop=True)
    return data_u


def date_sub(time1, time2, desc="get month"):
    """时间减法工具, get month得到时间1-时间2的月份, get year得到时间1-月份的时间2"""
    if desc == "get month":
        # 例子: 201201 - 201112 = 1
        year_sub = time1 // 100 - time2 // 100
        month_sub = time1 % 100 - time2 % 100
        return year_sub * 12 + month_sub
    else:
        # 例子: 201201 - 1 = 201112
        year_ = time1 // 100
        month_ = time1 % 100
        if month_ - time2 > 0:
            return time1 - time2
        else:
            return (year_ - 1) * 100 + 12 + month_ - time2


def num_average(num, n):
    """数字平均工具, 将一个数均匀分成n等份, 且每个数都依旧是整数, 如数字3分成2分, 则结果为[1, 2]"""
    n_example = num // n
    result = np.array([n_example] * n)
    add_1 = num - n_example * n
    result[-add_1:] += 1
    return result.tolist()


def data_average(data_u: pd.DataFrame):
    """
    :param data_u: pd数据集
    :return: data_all: 按月平均后的pd数据
    """
    data_e = data_u.copy(deep=True)
    source_title_set = set(data_e.loc[:, "Source Title"])

    # 将每个期刊的论文, 都均匀到月份, just this, 复杂点在于, 这个只是更改论文的出刊日期
    source_title_data = list()
    for source_title in source_title_set:
        data_one = data_e[data_e["Source Title"] == source_title]
        data_one = data_one.sort_values(by=["Publication Year", "DOI"], axis=0, ascending=True,
                                        inplace=False, ignore_index=True)
        date_paper_num = data_one["Publication Year"].value_counts(normalize=False, sort=False)
        # 对第一个时间, 向前延伸
        date_begin_process = list(date_paper_num.index[:2])
        month_sub = date_sub(date_begin_process[1], date_begin_process[0], "get month")
        date_begin = date_sub(date_begin_process[0], month_sub, "get time")
        date_paper_num.loc[date_begin] = 0
        date_paper_num = date_paper_num.reset_index()
        date_paper_num.sort_values(by="index", axis=0, inplace=True, ignore_index=True)

        # 建立时间-数据量对应的字典
        date_paper_num_dict = dict()
        date_paper_num_dict.setdefault("date", list())
        date_paper_num_dict.setdefault("paper num", list())
        for i in range(len(date_paper_num)-1):
            date1, paper_num1 = date_paper_num.loc[i]
            date2, paper_num2 = date_paper_num.loc[i+1]
            date_interval = date_sub(date2, date1, "get month")
            new_date = [date_sub(date2, i, "get time") for i in range(date_interval)][::-1]
            if paper_num2 >= date_interval:
                paper_num_average = num_average(paper_num2, date_interval)
                date_paper_num_dict["date"].extend(new_date)
                date_paper_num_dict["paper num"].extend(paper_num_average)
            else:
                date_paper_num_dict["date"].extend(new_date[-paper_num2:])
                date_paper_num_dict["paper num"].extend([1]*paper_num2)
        # 更换时间
        record = 0
        label_ = 0
        for i in range(len(data_one)):
            data_one.loc[i, "Publication Year"] = date_paper_num_dict["date"][label_]
            record += 1
            if record == date_paper_num_dict["paper num"][label_]:
                record = 0
                label_ += 1
        source_title_data.append(data_one)

    data_all = pd.concat(source_title_data, axis=0, ignore_index=True)
    data_all.sort_values(by=["Publication Year", "DOI"], axis=0, inplace=True, ignore_index=True)
    return data_all


"""
关键词处理代码, 对应于key_preprocess()
"""


def simi_get(abstract_, keyword_):
    """摘要提取与关键词相似度最高的3个词工具"""
    abs_ = remove_stopwords(abstract_)
    abs_ = nlp(abs_)
    key_ = [nlp(val) for val in keyword_]
    simi = dict()
    for nounc in abs_.noun_chunks:
        simi.setdefault(nounc.text, list())
        for val in key_:
            simi[nounc.text].append(val.similarity(nounc))
    simi = {key: max(value) for key, value in simi.items()}
    simi = sorted(simi.items(), key=lambda x: x[1], reverse=True)
    simi = [val[0] for val in simi]
    if len(simi) > 3:
        return simi[:3]
    else:
        return simi


def word_to_(w_: str):
    """分隔符转换工具: list_hello --> list hello"""
    return " ".join(w_.split("_"))


def word_process_spacy(words_, merge_=None, dict_=None):
    """关键词 摘要处理工具"""
    pattern_ = re.compile(r'x+', re.I)  # 匹配字符串是否有x | X

    if isinstance(words_, str):  # 对摘要的处理
        words_list = list()
        ind_all = list()
        doc_ = nlp(words_)
        for sent in doc_.sents:
            for word in sent:
                if "".join(set(re.sub(pattern_, "", word.shape_))) in ["", "_", "."]:
                    if (not word.is_stop) and (word.text not in ["_", "."]):  # 判断是否为停用词 | 常用词
                        word = word.lemma_.lower()
                        words_list.append(word)  # 返回词性还原后的词, 并小写
                        ind_all.extend(dict_.get(word, []))  # 筛选摘要中存在词的关键词
            words_list.append("#")  # 为后面分句做标识符
        words_list = words_list[:-1]  # 去掉最后一个没用的分句标识符

        ind_all = list(set(ind_all))
        merge_ = np.array(merge_)[ind_all]

        words = " ".join(words_list)
        for val in merge_:
            pattern_key = re.compile(word_to_(val), flags=re.I)
            words = re.sub(pattern_key, val, words)
        words_list = words.split(" ")
        return words_list

    elif isinstance(words_, list):  # 对关键词的处理
        words_list = list()
        for words in words_:
            word_ = nlp(words)
            word_temp = list()
            for word in word_:
                if "".join(set(re.sub(pattern_, "", word.shape_))) in ["", "_", "."]:
                    if (not word.is_stop) and (word.text not in ["_", "."]):  # 判断是否为停用词 | 常用词
                        word_temp.append(word.lemma_.lower())  # 返回词性还原后的词, 并小写
            if word_temp:  # 如果为空, 则放弃该关键词
                words_list.append("_".join(word_temp))
        return words_list
    else:
        return "输入有误, 请输入字符串或字符串列表!!!"


"""
分段线性法工具代码，对应于get_time_line()
"""


def topk(list_, k):
    """求列表中最大的前k个数字和下标"""
    list_ = np.array(list_)
    if len(list_) < k:
        k = len(list_)
    max_k = list()
    for i in range(k):
        max_ = np.max(list_)
        ind_ = np.argmax(list_)
        max_k.append([max_, ind_])
        list_[ind_] = -1
    return max_k


def get_point(data_line_):
    """获取折线图转折点候选词"""
    begin = data_line_[0]
    end = data_line_[-1]
    k = (end-begin) / (len(data_line_)-1)
    point_distance = list()
    for ind, val in enumerate(data_line_):
        point_distance.append(np.abs(val-k*ind-begin)/np.sqrt(1+k**2))
    max_k = topk(point_distance, 5)

    return max_k


def get_key_time(data_, list_, pre_index, limit_=3):
    """获取折线图转折点"""
    if len(data_) > 4:
        max_k = get_point(data_)
        for i in range(len(max_k)):  # 当对于获取的多个最大值点, 有一个满足下面要求, 则足以
            distance, index = max_k[i]
            if distance > 10:
                if index+1 > limit_ and len(data_)-index > limit_:
                    list_.append(pre_index+index)
                    get_key_time(data_[:index+1], list_, pre_index, limit_)
                    get_key_time(data_[index:], list_, pre_index+index, limit_)
                    break


"""
关键词摘要分段聚集工具代码，对应于time_gather()
"""


def high_pre_word(num_max=100):
    """输出整个摘要集中词频最大的前num_max个词"""
    dict_ = json_load("./data/time_gather/abstract_gather.json")
    abstract_pre = dict()
    abstract = list()
    for _, value in dict_.items():
        abstract.extend([va for val in value["abstract"] for va in val])
    for word in abstract:
        abstract_pre.setdefault(word, 0)
        abstract_pre[word] += 1
    abstract_pre = sorted(abstract_pre.items(), key=lambda x: x[1], reverse=True)
    abstract_pre = [word for word, pre in abstract_pre[:num_max]]
    json_save(abstract_pre, "./extra/extra/high_pre_word.json")


def abstract_filter(abstract_):
    """对摘要过滤摘要高频词"""
    high_pre_word(100)
    stop_ = json_load("./extra/extra/high_pre_word.json")
    abs_ = abstract_.split(";")
    abs_filter = list()
    for val in abs_:
        if val not in stop_:
            abs_filter.append(val)
    return abs_filter


"""
图模型建立工具，对应于build_graph()
"""


def word2vec_(sentences, keyword, thresh, time_, contrast=False):
    """word2vec代码pytorch版本，主要是因为简单掉包版本对gpu不支持，所以需要pytorch调用gpu
    详细链接：https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/docs/chapter10_natural-language-processing/10.3_word2vec-pytorch.md"""
    if contrast:
        sent_ = [len(val) for val in sentences]
        json_save(sent_, "./extra/skip_gram/二次采样前的句子长度情况.json")

    counter = Counter([word for sent in sentences for word in sent])  # token
    counter = dict(filter(lambda x: x[1] > 3, counter.items()))  # 剔除词频<=3的词

    ind_to_token = list(counter.keys())
    token_to_ind = {token: ind for ind, token in enumerate(ind_to_token)}

    ind_dataset = [[token_to_ind[word] for word in sent if word in ind_to_token] for sent in sentences]
    num_token = sum([len(sent) for sent in ind_dataset])

    # 二次采样
    def discard(ind):
        return random.uniform(0, 1) > 1 - math.sqrt(1e-4 / (counter[ind_to_token[ind]] / num_token))

    sub_ind_dataset = [[word for word in sent if discard(word)] for sent in ind_dataset]  # 二次采样结果
    if contrast:
        sent_ = [len(val) for val in sub_ind_dataset]
        json_save(sent_, "./extra/skip_gram/二次采样后的句子长度情况.json")

    # 提取中心词和背景词
    def get_centers_and_contexts(dataset, max_window_size):
        centers, contexts = [], []
        for sent in dataset:
            if len(sent) < 2:
                continue
            centers.extend(sent)
            for center_i in range(len(sent)):
                # window_size = random.randint(1, max_window_size)
                indices = list(
                    range(max(0, center_i - max_window_size), min(len(sent), center_i + 1 + max_window_size)))
                indices.remove(center_i)
                contexts.append([ind for ind in indices])
        return centers, contexts

    all_centers, all_contexts = get_centers_and_contexts(sub_ind_dataset, 3)  # 设置采样窗口为3

    # 负采样, 设置噪声
    def get_negatives(all_contexts_, sampling_weights_, k):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights_)))
        for contexts in all_contexts_:
            negatives = []
            while len(negatives) < len(contexts) * k:
                if i == len(neg_candidates):
                    i, neg_candidates = 0, random.choices(population, sampling_weights_, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    sampling_weights = [counter[w] ** 0.75 for w in ind_to_token]
    all_negatives = get_negatives(all_contexts, sampling_weights, 2)

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, ind):
            return self.centers[ind], self.contexts[ind], self.negatives[ind]

        def __len__(self):
            return len(self.centers)

    def batchify(data):
        max_len = max(len(c) + len(n) for _, c, n in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(
            labels)

    batch_size = 512
    num_workers = 0 if sys.platform.startswith("win32") else 8

    dataset = MyDataset(all_centers, all_contexts, all_negatives)
    data_iter = data_use.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers)

    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    class SigmoidBinaryCrossEntropyLoss(nn.Module):
        def __init__(self):
            super(SigmoidBinaryCrossEntropyLoss, self).__init__()

        def forward(self, inputs, targets, mask=None):
            inputs, targets, mask = inputs.float(), targets.float(), mask.float()
            res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
            return res.mean(dim=1)

    loss = SigmoidBinaryCrossEntropyLoss()

    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(ind_to_token), embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(ind_to_token), embedding_dim=embed_size))

    def train(net_, lr, num_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"train on {device}")
        net_ = net_.to(device)
        optimizer = torch.optim.Adam(net_.parameters(), lr=lr)
        for epoch in range(num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in data_iter:
                center, context_negative, mask, label = [d.to(device) for d in batch]
                pred = skip_gram(center, context_negative, net_[0], net_[1])

                l = (loss(pred.view(label.shape), label, mask) * mask.shape[1] / mask.float().sum(dim=1)).mean()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1
            # print("epoch %d, loss %.2f, time %.2fs" % (epoch + 1, l_sum / n, time.time() - start))

    train(net, 0.01, 100)

    g = nx.Graph()
    W = net[0].weight.data
    keyword = [val for val in keyword if val in ind_to_token]
    for ind, key_1 in tqdm(enumerate(keyword), total=len(keyword), desc="图g边生成", ncols=100):
        k1_vector = W[token_to_ind[key_1]]
        cos = torch.matmul(W, k1_vector) / (
                torch.sum(W * W, dim=1) * torch.sum(k1_vector * k1_vector) + 1e-9).sqrt()
        cos = cos.cpu().numpy()
        for key_2 in keyword[ind + 1:]:
            cos_ = cos[token_to_ind[key_2]]
            if cos_ > thresh:
                g.add_edge(key_1, key_2, weight=cos_)
    torch.save(net, "./data/build_graph/word2vec/model/" + time_ + ".model")
    pickle_save(g, "./data/build_graph/word2vec/graph/" + time_ + ".graph")
    json_save(token_to_ind, "./data/build_graph/word2vec/ind/" + time_ + ".json")


def node2vec_(g_, name):
    """node2vec代码pytorch版本，使用这个而不是直接调包主要是因为调包代码写的太烂（虽然这个也很拉），极度吃内存导致容易运行失败"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matrix_ = nx.edges(g_)
    g_node = list(g_.nodes())
    node_num = {node_: ind for ind, node_ in enumerate(g_node)}
    edges_ = [[], []]
    for node1, node2 in matrix_:
        edges_[0].append(node_num[node1])
        edges_[1].append(node_num[node2])
    edges_ = np.array(edges_)
    matrix_ = torch.from_numpy(edges_).type(torch.LongTensor)

    model = Node2Vec(matrix_, embedding_dim=128, walk_length=20,
                     context_size=3, walks_per_node=50, num_negative_samples=15,
                     sparse=True, q=0.25).to(device)
    model = torch.nn.DataParallel(model)

    loader = model.module.loader(batch_size=128, shuffle=True, num_workers=8)
    optimizer = torch.optim.SparseAdam(model.module.parameters(), lr=0.01)

    def train():
        model.module.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.module.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(100):
        train()

    torch.save(model, "./data/build_graph/node2vec/model/"+name+".model")
    model.eval()
    vector_ = model(torch.arange(len(g_node), device=device)).cpu().detach().numpy()
    json_save(node_num, "./data/build_graph/node2vec/node_num/"+name+"_node_num.json")

    return node_num, vector_


def text_graph(text, window=2):
    """
    :param text: 单文本段落
    :param window: 设置共现窗口大小
    :return: 以该文本段落构建的词图网络，以词作节点，词相邻则作边，添加属性：weight：词相邻次数；distance=1/weight：自定义词距离
    """
    edge_dict = dict()
    g = nx.Graph()
    # 创建词图网络，以词作节点，词相邻则作边，添加属性：weight：词相邻次数；distance=1/weight：自定义词距离
    for i, word in enumerate(text):
        word_a = word
        for word_b in text[i+1: i+window+1]:
            edge_ = frozenset([word_a, word_b])
            edge_dict.setdefault(edge_, 0)
            edge_dict[edge_] += 1
            # 利用graph填充重复边覆盖的性质, 个人感觉, 如果害怕还是耗时长, 可以放到循坏外一次性+边
            g.add_edge(word_a, word_b, weight=edge_dict[edge_], distance=1/edge_dict[edge_])
    return g


def path_score(g, node1, node2, weight="distance"):
    """
    :param g: 图网络
    :param node1: 路径起点
    :param node2: 路径终点
    :param weight: 最短路判定标准
    :return: 返回两点最短路的 路径列表 和 得分
    """
    # 获取图网络两点的最短路径
    path = nx.shortest_path(g, source=node1, target=node2, weight=weight)
    # 获取每段路径的边
    pairs = list(zip(path, path[1:] + path[:-1]))[:-1]
    score = 0
    for pair in pairs:
        score += 1 / g[pair[0]][pair[1]]["weight"]
    return path, 1 / score


def sub_graph(g, sub_nodes):
    """
    :param g: 图网络
    :param sub_nodes: 用于做子图的点
    :return: 如下规则做的子图
    """
    # sub_nodes = word_filter(sub_nodes)
    common_nodes = list(set(sub_nodes).intersection(set(list(g.nodes))))
    pairs = list(combinations(common_nodes, 2))
    edge_list = []
    for pair in pairs:
        path, score = path_score(g, pair[0], pair[1])
        # 这个东西是这样判断的: 如果两个节点是直连,则子图要这条边
        # 如果两个节点不是直连,则如果它们的最短路中不包含子图点,也要这条边
        # 总之,对于子图,不要那些最短路包含子图点的新建边
        judge = (list(set(path[1: -1]).intersection(set(sub_nodes))) == []) | (path[1: -1] == [])
        if judge:
            edge_list.append((pair[0], pair[1], {'weight': score}))
    sub_g = nx.Graph(edge_list)
    return sub_g


def add_attr(g, node_comm=None):
    """
    :param g: 图网络
    :param node_comm: 社区划分结果
    :return: 节点添加了degree betweenness eigenvector closeness centrality community 属性, 边添加了distance属性
    """
    def data_normal(dict_):
        """
        :param dict_: 需对值归一化的数据字典
        :return: 归一化后的数据字典
        """
        min_ = min(dict_.items(), key=lambda x: x[1])[1]
        max_ = max(dict_.items(), key=lambda x: x[1])[1]
        try:
            res = dict((k, (v - min_) / (max_ - min_)) for k, v in dict_.items())
            return res
        except Exception as e:
            return dict_

    degree = data_normal(degree_centrality(g))
    betweenness = data_normal(betweenness_centrality(g))
    eigenvector = data_normal(eigenvector_centrality(g, max_iter=1000))
    closeness = data_normal(closeness_centrality(g))
    # 字典性质的三性相加
    counter = Counter()
    for centrality in [closeness, degree, eigenvector]:
        counter.update(centrality)
    for node in g:
        g.nodes[node]['degree'] = degree[node]
        g.nodes[node]['betweenness'] = betweenness[node]
        g.nodes[node]['eigenvector'] = eigenvector[node]
        g.nodes[node]['closeness'] = closeness[node]
        g.nodes[node]['centrality'] = counter[node]
        if node_comm is not None:
            g.nodes[node]['community'] = node_comm[node]
    for edge in g.edges:
        g[edge[0]][edge[1]]['distance'] = 1 / g[edge[0]][edge[1]]['weight']

    return g


def final_graph(g_list, title="test", export=False):
    """
    :param g_list: 时序图网络列表
    :param title: 终代图网络名
    :param export: 是否输出存储
    :return: 终代时序图网络
    """
    final_g = nx.Graph()
    # 图网络构建
    edge_dict = dict()
    for g_one in g_list:
        for edge in list(g_one.edges):
            node_0 = edge[0]
            node_1 = edge[1]
            w = g_one[node_0][node_1]['weight']
            edge_ = frozenset([node_0, node_1])
            edge_dict.setdefault(edge_, 0)
            edge_dict[edge_] += w
            final_g.add_edge(node_0, node_1, weight=edge_dict[edge_])
    # 图网络添加属性
    final_g = add_attr(final_g)
    if export:
        pickle_save(final_g, title)
    return final_g


def use_model(ab, key, thr, t, model):
    """模型选择处理"""
    if model == "word2vec":
        word2vec_(ab, key, thr, t)
    else:
        g_list = list()
        for ab_ in tqdm(ab):
            g_abstract = text_graph(ab_)
            topics_sub_abstract = sub_graph(g_abstract, key)
            g_list.append(topics_sub_abstract)
        if model == "node2vec":
            final_g = final_graph(g_list, "./data/build_graph/node2vec/graph/" + t + ".graph", True)
            node2vec_(final_g, t)
        elif model == "jaccard":
            final_graph(g_list, "./data/build_graph/jaccard/graph/" + t + ".graph", True)


"""
社区核心词获取工具，对应于get_mix_g_and_key_word()
"""


def z_score(g: nx.Graph(), is_norm=False):
    """z得分公式"""
    nodes = g.nodes()
    m = len(nodes)
    b = 0
    q = 0
    n = dict()
    for node in nodes:
        node_nei = g.neighbors(node)
        weight_sum = 0
        for nei in node_nei:
            edge_weight = g.get_edge_data(node, nei)["weight"]
            weight_sum += edge_weight
        n[node] = weight_sum
        b += weight_sum
        q += np.power(weight_sum, 2)

    z = {node: (n[node] - b/m) / np.power((q/m - np.power(b/m, 2)), 0.5) for node in nodes}
    if is_norm:
        z_value = z.values()
        z = {node: (value-min(z_value))/(max(z_value)-min(z_value)) for node, value in z.items()}
    return z


"""
对应社区词相似度工具，对应于get_comm_word_sims()
"""


def cos_sim(vector1, vector2):
    """余弦相似度计算公式"""
    return np.divide(np.dot(vector1, vector2), np.linalg.norm(vector1) * np.linalg.norm(vector2))


"""
杂项处理代码，如相似度词典制作
"""


def synonym_make():
    """处理相似度词典文件，转换成可读可查找的dict字典文件"""
    ind_judge = 0
    words_dict = dict()
    with open("../extra/synonym/Collins2020.txt", "r") as f:
        for line in tqdm(f.readlines()):
            ind_judge += 1
            line = line.strip("\n").strip(" ")
            if ind_judge == 1:
                if line[0] == "<":
                    ind_judge = 2
                    continue
                key_ = line
                words_dict.setdefault(key_, list())
            elif ind_judge == 2:
                try:
                    soup = BeautifulSoup(line, "html.parser")
                except MarkupResemblesLocatorWarning as e:
                    continue
                orth = soup.find_all("div", class_="form type-syn")
                orth = [val.find_next("span", class_="orth").text.strip() for val in orth]

                li = soup.find_all("div", class_="columns-block")
                li = [vals.text.strip() for val in li for vals in val.find_all_next("li")]
                data_ = orth + li
                words_dict[key_].extend(data_)
            elif ind_judge == 3:
                ind_judge = 0

    json_save(words_dict, "../json/synonym.json")


def find_year(year, time_line):
    """查看某条论文数据的出版年月，属于哪个时间段"""
    year = int(year)
    return_year = time_line[-1]
    for ind, year_ in enumerate(time_line):
        if year <= year_:
            return_year = year_
            break
    return str(return_year)


def graphml_comm_evo():
    """社区演绎图制作, 拿着生成的comm_evo.graphml去制作吧"""
    g = nx.Graph()
    name_comm = json_load(f"./data/calc_relevance/word2vec/degree/name_comm.json")
    labels = ["comm8-6", "comm1-6", "mix3-6.5", "comm10-7", "comm6-7", "comm4-7"]
    node_all = [name_comm[label] for label in labels]
    nodes = list()
    edges = list()
    for ind, node_one in enumerate(node_all):
        for node in node_one:
            label = labels[ind]
            nodes.append((node, {"comm_group": label,
                                 "group": label.split("-")[1]}))
            nodes.append((label, {"comm_group": label,
                                  "group": label.split("-")[1]}))

    center = labels[2]
    for edge in labels[:2] + labels[3:]:
        weight = len(set(name_comm[edge]) & set(name_comm[center]))
        edges.append((edge, center, {"weight": weight}))
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.write_graphml(g, "./graphml/comm_evo.graphml")


def graphml_word_evo():
    """关键词演绎图制作, 拿着生成的comm_evo.graphml去制作吧"""
    g = nx.Graph()
    name_comm = json_load(f"./data/calc_relevance/word2vec/degree/name_comm.json")
    comm_comm_link = pickle_load(f"./data/calc_relevance/word2vec/degree/comm_comm_link.pickle")
    labels = ["comm8-6", "comm1-6", "comm10-7", "comm6-7", "comm4-7"]

    node_all = [name_comm[label] for label in labels]
    nodes = list()
    edges = list()
    for ind, node_one in enumerate(node_all):
        for node in node_one:
            label = labels[ind]
            nodes.append((node, {"comm_group": label,
                                 "group": label.split("-")[1]}))

    for label in labels[2:]:
        comm = comm_comm_link[label]
        for node, max_node in comm.items():
            edges.append((node, max_node[0][0], {"weight": max_node[0][1]}))
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.write_graphml(g, "./graphml/word_evo.graphml")
