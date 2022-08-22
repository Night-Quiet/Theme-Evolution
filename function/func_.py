from func_no_care import *
import torch
from function.draw_ import draw_time_line
from tqdm import tqdm
import networkx as nx
from keybert import KeyBERT
import matplotlib.pyplot as plt
import spacy
from function.twoNodeConnect_louvain import twoConnect_louvain as TC
import os


# 总设置
# 加载英文数据处理
nlp = spacy.load("en_core_web_lg")
kw_model = KeyBERT()


def data_preprocess():
    """
    :return: 数据预处理, 获得干净的数据集, 并按出版年份排序
    """
    if not os.path.exists("./data/data_preprocess/paper.csv"):
        data_concat()
    if not os.path.exists("./data/data_preprocess/paper_update.csv"):
        data = pd.read_csv("./data/data_preprocess/paper.csv", index_col=0, low_memory=False)
        data = data_filter(data)
        data.sort_values(["Publication Year"], axis=0, inplace=True, ignore_index=True)
        data.to_csv("./data/data_preprocess/paper_update.csv")
    if not os.path.exists("./data/data_preprocess/paper_end.csv"):
        data = pd.read_csv("./data/data_preprocess/paper_update.csv", index_col=0, low_memory=False)
        data = data_average(data)
        data.to_csv("./data/data_preprocess/paper_end.csv")
    print("******** data_preprocess() 完成 ********\n")


def key_preprocess(model_set="keyword"):
    """
    :param model_set: 模型选择, 如为"simi", 则是提取摘要中, 与关键词相似度最高的前3个词补充keyword,
                      如为"keyword", 则是直接在摘要中提取3个摘要关键词补充带keyword
    :return: 得到补充、去停用词、去科学常用词的关键词, 得到去停用词、关键词保留的摘要，
             此时还是csv格式，即还是对单条数据操作
    """
    # 数据获取
    dtype = dict()
    usecols = ["Authors", "Author Full Names", "Article Title", "Author Keywords",
               "Keywords Plus", "Abstract", "Publication Year"]
    for col in usecols:
        dtype[col] = "str"
    dtype["Publication Year"] = "int"
    data = pd.read_csv("./data/data_preprocess/paper_end.csv", low_memory=False, usecols=usecols, sep=",", dtype=dtype)
    data = data.fillna(value="", inplace=False)
    data.loc[:, "keyword_plus"] = None
    data.loc[:, "title_abstract"] = None

    # 关键词处理
    keyword_plus_all = list()
    paper_stopword = set(json_load("./data/key_preprocess/paper_stopword.json"))
    for i in tqdm(list(data.index), ncols=100, total=len(list(data.index))):
        # 对关键词处理, 并预备将关键词加入分词规则
        keyword_list = word_process_spacy(data.loc[i, "Author Keywords"].split("; "))
        keyword_plus_list = word_process_spacy(data.loc[i, "Keywords Plus"].split("; "))
        keyword_plus = keyword_list

        # 对摘要, 标题处理, 进行关键词补充
        title = data.loc[i, "Article Title"]
        abstract = data.loc[i, "Abstract"]
        abs_ = title + abstract
        if model_set == "keyword":
            keyword_add = kw_model.extract_keywords(abs_, keyphrase_ngram_range=(1, 2), top_n=3)
            keyword_add = [val[0] for val in keyword_add]
        else:
            keyword_add = simi_get(abs_, keyword_list)

        keyword_plus += keyword_add
        keyword_plus = list(set(keyword_plus) - paper_stopword)
        data.loc[i, "keyword_plus"] = ";".join(keyword_plus)
        keyword_plus_all.extend(keyword_plus)

    keyword_plus_all = list(set(keyword_plus_all))

    # 做词检验, 做词典, 将每个词对应一个编号列表, 使得到时候查词的时候可以知道词在keyword_plus_all的哪些位置中
    word_dict = dict()
    for ind, val_ in enumerate(keyword_plus_all):
        word_split = val_.split("_")
        for _ in word_split:
            word_dict.setdefault(_, list())
            word_dict[_].append(ind)

    # 标题摘要处理
    for i in tqdm(list(data.index), ncols=100, total=len(list(data.index))):
        title = word_process_spacy(data.loc[i, "Article Title"], keyword_plus_all, word_dict)
        abstract = word_process_spacy(data.loc[i, "Abstract"], keyword_plus_all, word_dict)
        data.loc[i, "title_abstract"] = ";".join(title + abstract)

    data.to_csv("./data/key_preprocess/paper_word.csv", index=False)
    print("******** key_preprocess() 完成 ********\n")


def get_time_line(limit_=3, remove_big=True, combine=0):
    """
    :param limit_: 限制时间序列分段过程中, 每段至少有limit_个时间点
    :param remove_big: 设置是否删除每年1月份的关键词
    :param combine: 设置是否对月份合并, 如为0则不合并, 如为>0数字则按顺序每combine个月合并一次
    :return: 关键转折时间的列表
    """
    data_ = pd.read_csv("./data/key_preprocess/paper_word.csv", skip_blank_lines=True)
    month_keyword = dict()
    for i in range(len(data_)):
        time_ = data_.loc[i, "Publication Year"]
        keyword = str(data_.loc[i, "keyword_plus"])
        if remove_big and str(time_)[-2:] == "01":
            continue
        month_keyword.setdefault(time_, list())
        month_keyword[time_].extend(keyword.split(";"))
    time_keyword_num = {key: len(set(value)) for key, value in month_keyword.items()}
    keyword_num = list(time_keyword_num.values())[12:-6]  # 删掉后面的数据, 数值不齐

    if combine > 0:  # 对数据进行按季度合并
        combine_judge = 0
        combine_time = 0
        combine_keyword = dict()
        for key, value in month_keyword.items():
            if combine_judge == 0:
                combine_judge += 1
                combine_time = key
                combine_keyword.setdefault(combine_time, list())
            else:
                combine_judge += 1
                combine_judge = np.divmod(combine_judge, combine)[1]
            combine_keyword[combine_time].extend(value)
        time_keyword_num = {key: len(set(value)) for key, value in combine_keyword.items()}
        keyword_num = list(time_keyword_num.values())[12//combine:-6//combine]  # 不要后面的数据, 数据不齐

    key_time = list()
    get_key_time(keyword_num, key_time, 0, limit_)
    key_time = np.array(key_time)+12
    key_time = key_time.tolist()
    key_time.extend([0, len(keyword_num) + 12 - 1])
    key_time = sorted(key_time, reverse=False)

    draw_time_line(key_time, time_keyword_num)  # 画基于时间分段的线形图

    key_time = np.array(list(time_keyword_num.keys()))[key_time]
    key_time = key_time.tolist()  # 折线图的转折点
    json_save(key_time, "./data/get_time_line/key_time.json")
    json_save(len(key_time), "./data/get_time_line/key_time_len.json")
    print(f"--------关键转折时间为: {key_time}")
    print("******** get_time_line() 完成 ********\n")


def time_gather():
    """
    :return: 将摘要、关键词按照时间分段聚集后的json文件
    """
    data_ = pd.read_csv("./data/key_preprocess/paper_word.csv", skip_blank_lines=True)
    key_time = json_load("./data/get_time_line/key_time.json")
    data_point = dict()
    data_point_abstract = dict()
    time_begin = key_time[0]
    for ind, time_ in enumerate(key_time[1:]):
        df = data_[data_["Publication Year"] < time_]
        df = df[df["Publication Year"] >= time_begin]
        time_begin = time_
        keyword_ = df.loc[:, "keyword_plus"]
        keyword_ = list(set([v for val in keyword_ for v in str(val).split(";")]))
        abstract_all = df.loc[:, "title_abstract"]

        abstract = list()  # 按句子分的摘要集
        abstract_ = list()  # 按摘要分的摘要集
        for one_abs in abstract_all:
            sent_all = list()
            for sent_ in one_abs.split(";#;"):
                sent_ = abstract_filter(sent_)
                abstract.append(sent_)
                sent_all.extend(sent_)
            abstract_.append(sent_all)

        data_point.setdefault("year" + str(ind), dict())
        data_point["year" + str(ind)]["keyword"] = keyword_
        data_point["year" + str(ind)]["abstract"] = abstract

        data_point_abstract.setdefault("year" + str(ind), dict())
        data_point_abstract["year" + str(ind)]["keyword"] = keyword_
        data_point_abstract["year" + str(ind)]["abstract"] = abstract_

    json_save(data_point, "./data/time_gather/sentence_gather.json")  # 按句子分
    json_save(data_point_abstract, "./data/time_gather/abstract_gather.json")  # 按摘要分
    print("******** time_gather() 完成 ********\n")


def build_model(time_, thresh=0.25, model_="word2vec", time_old=""):
    """
    :param time_: 确定该模型针对的时间段
    :param thresh: 模型建图时对边的过滤
    :param model_: 选择模型
    :param time_old: 确定上一个时间段，做连续时间段的融合模型
    :return: 图网络、图模型
    """
    if model_ == "word2vec":
        data_point = json_load("./data/time_gather/sentence_gather.json")
    else:
        data_point = json_load("./data/time_gather/abstract_gather.json")

    if time_ != "all":
        abstract_ = data_point[time_]["abstract"]
        keyword = data_point[time_]["keyword"]
    else:
        abstract_ = list()
        keyword = list()
        for year, data_ in data_point.items():
            abstract_.extend(data_["abstract"])
            keyword.extend(data_["keyword"])
    use_model(abstract_, keyword, thresh, time_, model_)

    # 做混合模型
    if time_old:
        abstract_old = data_point[time_old]["abstract"]
        keyword_old = data_point[time_old]["keyword"]
        abstract_.extend(abstract_old)
        keyword.extend(keyword_old)
        use_model(abstract_, keyword, thresh, time_old[:4]+str(int(time_old[4:])+0.5)+"_mix", model_)


def build_graph(model_="word2vec", thresh=0.25):
    """
    :param model_: 模型选择
    :param thresh: 图网络筛选边过滤阈值选择
    :return: 图网络 & 图模型
    """
    key_time_len = int(json_load("./data/get_time_line/key_time_len.json")) - 1
    build_model("all", thresh, model_)
    for ind in range(key_time_len):
        if ind == 0:
            time_ = "year" + str(ind)
            build_model(time_, thresh, model_)
        else:
            time_ = "year" + str(ind)
            time_old = "year" + str(ind-1)
            build_model(time_, thresh, model_, time_old)
    print("******** build_graph() 完成 ********\n")


def handle_g(model="word2vec", filter_set="degree"):
    """
    :param model: 模型设定
    :param filter_set: 核心节点选择指标设定
    :return:
    """
    key_time_len = int(json_load("./data/get_time_line/key_time_len.json")) - 1
    time_comm = dict()
    mix_comm = dict()
    old_g = nx.Graph()
    for i in tqdm(range(key_time_len)):
        g = pickle_load("./data/build_graph/"+model+"/graph/year" + str(i) + ".graph")
        time_comm.setdefault(i, list())
        for val in TC(g, "temp" + str(i)):
            g_sub = nx.Graph(nx.subgraph(g, val))
            if filter_set == "degree":
                filter_score = degree_centrality(g_sub)
                g_sub_node = [word for word, score in filter(lambda x: x[1] > 0.1, filter_score.items())]
            else:
                filter_score = z_score(g_sub)
                g_sub_node = [word for word, score in filter(lambda x: x[1] >= 2.5, filter_score.items())]
            if not g_sub_node:
                time_comm[i].append([sorted(filter_score.items(), key=lambda x: x[1], reverse=True)[0][0]])
            else:
                time_comm[i].append(g_sub_node)
        if i == 0:
            old_g = nx.Graph(g)
            continue
        else:
            mix_g = final_graph([old_g, g])
            mix_comm.setdefault(i, list())
            for val in TC(mix_g, "mix" + str(i)):
                mix_comm[i].append(val)
            old_g = nx.Graph(g)
    json_save(time_comm, f"./data/handle_g/{model}/{filter_set}_subject_words.json")
    json_save(mix_comm, f"./data/handle_g/{model}/{filter_set}_mix_subject_words.json")


def calc_relevance(model_="word2vec", filter_set="degree"):
    """
    :param model_: 模型设定
    :param filter_set: 核心节点选择指标设定
    :return:
    """
    time_comm = json_load(f"./data/handle_g/{model_}/{filter_set}_subject_words.json")
    mix_comm = json_load(f"./data/handle_g/{model_}/{filter_set}_mix_subject_words.json")
    time_comm_key = list(time_comm.keys())

    name_comm = dict()
    mix_comm_link = dict()

    nodes_with_words = list()
    nodes_no_words = list()
    edges = list()
    colors = dict()
    for i, j in zip(time_comm_key[:-1], time_comm_key[1:]):
        time_comm_i = time_comm[i]
        time_comm_j = time_comm[j]
        time_mix_k = mix_comm[j]
        for ind_i, comm_i_one in enumerate(time_comm_i):
            nodes_with_words.append((f"comm{ind_i}-{i}", {"label": f"comm{ind_i}-{i}",
                                                          "group": float(i), "z": str(i),
                                                          "score": len(comm_i_one), "comm_group": f"comm{ind_i}-{i}"}))
            nodes_no_words.append((f"comm{ind_i}-{i}", {"label": f"comm{ind_i}-{i}",
                                                        "group": float(i), "z": str(i),
                                                        "score": len(comm_i_one), "comm_group": f"comm{ind_i}-{i}"}))
            colors[f"comm{ind_i}-{i}"] = int(i)

            name_comm[f"comm{ind_i}-{i}"] = comm_i_one

            "------------尝试------------"
            for node in comm_i_one:
                nodes_with_words.append((f"{ind_i}-{node}", {"label": f"{ind_i}-{node}", "group": float(i),
                                                             "z": str(float(i)), "score": 1,
                                                             "comm_group": f"comm{ind_i}-{i}"}))
            "------------尝试------------"

            max_mix_len = 0
            max_mix_ind = 0
            for ind_k, mix_k_one in enumerate(time_mix_k):
                nodes_with_words.append((f"mix{ind_k}-{int(i)+0.5}",
                                         {"label": f"mix{ind_k}-{int(i)+0.5}", "group": float(i)+0.5,
                                          "z": str(float(i)+0.5), "score": len(mix_k_one),
                                          "comm_group": f"mix{ind_k}-{int(i)+0.5}"}))
                nodes_no_words.append((f"mix{ind_k}-{int(i) + 0.5}",
                                       {"label": f"mix{ind_k}-{int(i) + 0.5}", "group": float(i) + 0.5,
                                        "z": str(float(i) + 0.5), "score": len(mix_k_one),
                                        "comm_group": f"mix{ind_k}-{int(i) + 0.5}"}))
                colors[f"mix{ind_k}-{int(i)+0.5}"] = -1

                name_comm[f"mix{ind_k}-{int(i)+0.5}"] = mix_k_one

                mix_comm_link.setdefault(f"mix{ind_k}-{int(i)+0.5}", [[], []])
                comm_mix_intersect = set(comm_i_one) & set(mix_k_one)
                if len(comm_mix_intersect) > max_mix_len:
                    max_mix_len = len(comm_mix_intersect)
                    max_mix_ind = ind_k
            edges.append((f"comm{ind_i}-{i}", f"mix{max_mix_ind}-{int(i)+0.5}", {"weight": max_mix_len}))
            mix_comm_link[f"mix{max_mix_ind}-{int(i)+0.5}"][0].append(f"comm{ind_i}-{i}")
        for ind_j, comm_j_one in enumerate(time_comm_j):
            nodes_with_words.append((f"comm{ind_j}-{j}",
                                     {"label": f"comm{ind_j}-{j}", "group": float(j), "z": str(float(j)),
                                      "score": len(comm_j_one), "comm_group": f"comm{ind_j}-{j}"}))
            nodes_no_words.append((f"comm{ind_j}-{j}",
                                   {"label": f"comm{ind_j}-{j}", "group": float(j), "z": str(float(j)),
                                    "score": len(comm_j_one), "comm_group": f"comm{ind_j}-{j}"}))
            colors[f"comm{ind_j}-{j}"] = int(j)

            name_comm[f"comm{ind_j}-{j}"] = comm_j_one

            "------------尝试------------"
            for node in comm_j_one:
                nodes_with_words.append((f"{ind_j}-{node}",
                                         {"label": f"{ind_j}-{node}", "group": float(j), "z": str(j), "score": 1,
                                          "comm_group": f"comm{ind_j}-{j}"}))
            "------------尝试------------"

            max_mix_len = 0
            max_mix_ind = 0
            for ind_k, mix_k_one in enumerate(time_mix_k):
                comm_mix_intersect = set(comm_j_one) & set(mix_k_one)
                if len(comm_mix_intersect) > max_mix_len:
                    max_mix_len = len(comm_mix_intersect)
                    max_mix_ind = ind_k
            edges.append((f"comm{ind_j}-{j}", f"mix{max_mix_ind}-{int(i)+0.5}", {"weight": max_mix_len}))
            mix_comm_link[f"mix{max_mix_ind}-{int(i)+0.5}"][1].append(f"comm{ind_j}-{j}")

    json_save(mix_comm_link, f"./data/calc_relevance/{model_}/{filter_set}/mix_comm_link.json")
    json_save(nodes_with_words, f"./data/calc_relevance/{model_}/{filter_set}/nodes_with_words.json")
    json_save(nodes_no_words, f"./data/calc_relevance/{model_}/{filter_set}/nodes_no_words.json")
    json_save(edges, f"./data/calc_relevance/{model_}/{filter_set}/edges.json")
    json_save(name_comm, f"./data/calc_relevance/{model_}/{filter_set}/name_comm.json")

    g_with_words = nx.Graph()
    g_with_words.add_nodes_from(nodes_with_words)
    g_with_words.add_edges_from(edges)
    nx.write_graphml(g_with_words, f"./data/calc_relevance/{model_}/{filter_set}/{model_}.graphml")
    nx.write_gexf(g_with_words, f"./data/calc_relevance/{model_}/{filter_set}/{model_}.gexf")

    g_no_word = nx.Graph()
    g_no_word.add_nodes_from(nodes_no_words)
    g_no_word.add_edges_from(edges)
    edge_size = {i[0:2]: i[2]['weight'] for i in g_no_word.edges(data=True)}
    pos_node = dict()
    nodes_size = dict()
    for i in g_no_word.nodes(data=True):
        if i[0][:3] == "mix":
            nodes_size[i[0]] = i[1]["score"] * 20
        else:
            nodes_size[i[0]] = i[1]["score"] * 200
        pos_node.setdefault(i[1]["label"], list()).append(i[0])

    pos = nx.multipartite_layout(g_no_word, subset_key="group")
    pos_ = dict()

    for key, value in pos.items():
        pos_[key] = np.array([value[0] * 300, value[1] * 100])

    plt.figure(figsize=(300, 100))
    nx.draw(g_no_word, pos=pos_, nodelist=list(colors.keys()), edgelist=list(g_no_word.edges(data=True)),
            node_color=list(colors.values()), node_size=[nodes_size[node] for node in colors.keys()],
            cmap=plt.cm.Set1, with_labels=True, width=list(edge_size.values()), font_size=24)
    plt.axis("equal")
    plt.savefig(f"./data/calc_relevance/{model_}/{filter_set}/社区相似度并行图.pdf")
    plt.savefig(f"./data/calc_relevance/{model_}/{filter_set}/社区相似度并行图.svg")
    plt.savefig(f"./data/calc_relevance/{model_}/{filter_set}/社区相似度并行图.jpg")
    pickle_save(g_no_word, f"./data/calc_relevance/{model_}/{filter_set}/g_no_word.graph")
    comm_comm_link = dict()
    for comm, (link1, link2) in mix_comm_link.items():
        if model_ == "word2vec":
            model = torch.load(f"./data/build_graph/{model_}/model/year{comm.split('-')[-1]}_mix.model")
            w = model[0].weight.data
            token_to_ind = json_load(f"./data/build_graph/{model_}/ind/year{comm.split('-')[-1]}_mix.json")
        elif model_ == "node2vec":
            model = torch.load(f"./data/build_graph/{model_}/model/year{comm.split('-')[-1]}_mix.model")
            node_num = json_load(f"./data/build_graph/{model_}/node_num/year{comm.split('-')[-1]}_mix_node_num.json")
            vectors = model(torch.arange(len(node_num), device="cuda")).cpu().detach().numpy()
        else:
            g_ = pickle_load(f"./data/build_graph/{model_}/graph/all.graph")
        if link1 and link2:
            for link2_one in link2:
                comm_comm_link.setdefault(link2_one, dict())
                for word2 in name_comm[link2_one]:
                    comm_comm_link[link2_one].setdefault(word2, list())
                    if model_ == "word2vec":
                        word2_vector = w[token_to_ind[word2]]
                        cos = torch.matmul(w, word2_vector) / (torch.sum(w*w, dim=1) *
                                                               torch.sum(word2_vector*word2_vector)+1e-9).sqrt()
                        cos = cos.cpu().numpy()
                    max_sim = 0
                    max_word = ""
                    for link1_one in link1:
                        for word1 in name_comm[link1_one]:
                            if model_ == "word2vec":
                                sims = cos[token_to_ind[word1]]
                            elif model_ == "node2vec":
                                sims = cos_sim(vectors[node_num[word2]], vectors[node_num[word1]])
                            else:
                                short_len = nx.shortest_path_length(g_, source=word2, target=word1)
                                if short_len > 2:
                                    sims = 0
                                elif short_len == 2:
                                    val1_n = set([val for val in g_.neighbors(word2)])
                                    val2_n = set([val for val in g_.neighbors(word1)])
                                    sims = 0.5 * len(val1_n & val2_n) / len(val1_n | val2_n)
                                else:
                                    val1_n = set([val for val in g_.neighbors(word2)])
                                    val2_n = set([val for val in g_.neighbors(word1)])
                                    sims = len(val1_n & val2_n) / len(val1_n | val2_n)
                            if sims > max_sim:
                                max_sim = sims
                                max_word = word1
                    comm_comm_link[link2_one][word2].append([max_word, max_sim])
                    comm_comm_link[link2_one][word2] = sorted(comm_comm_link[link2_one][word2], key=lambda x: x[1],
                                                              reverse=True)

    pickle_save(comm_comm_link, f"./data/calc_relevance/{model_}/{filter_set}/comm_comm_link.pickle")
    print("******** calc_relevance() 完成 ********\n")


def comm_evo(model_, filter_set):
    # 新生
    new_born = dict()
    # 消亡
    die_out = dict()
    # 继承
    inherit = dict()
    # 分裂
    divide = dict()
    # 融合
    merge = dict()
    # 孤立
    isolate = dict()
    mix_comm_link = json_load(f"./data/calc_relevance/{model_}/{filter_set}/mix_comm_link.json")
    # mix_comm_link = {mix(混合社区号)-(混合社区年份): [[]老社区集, []新社区集]}
    new_die_isolate_judge = dict()
    years = set()
    for mix, (old_comm, new_comm) in mix_comm_link.items():
        year = mix.split("-")[1]
        year_old = int(float(year))
        year_new = int(float(year))+1
        years.add(year_old)
        years.add(year_new)
        # 新生: 与前无关, 与后有关
        # 消亡: 与前有关, 与后无关
        # 孤立: 与前无关, 与后无关
        if not old_comm and new_comm:
            new_die_isolate_judge.setdefault(year_new, [[], []])
            new_die_isolate_judge[year_new][0].extend(new_comm)  # 该年份与前无关社区
        if old_comm and not new_comm:
            new_die_isolate_judge.setdefault(year_old, [[], []])
            new_die_isolate_judge[year_old][1].extend(old_comm)  # 该年份与后无关社区
        # 继承
        if len(old_comm) == 1 and len(new_comm) == 1:
            inherit.setdefault(year_old, list()).extend(old_comm)
            inherit.setdefault(year_new, list()).extend(new_comm)
        # 分裂
        if len(old_comm) == 1 and len(new_comm) > 1:
            divide.setdefault(year_old, list()).extend(old_comm)
        # 融合
        if len(old_comm) > 1 and len(new_comm) == 1:
            merge.setdefault(year_new, list()).extend(new_comm)

    # 新生: 与前无关, 与后有关
    # 消亡: 与前有关, 与后无关
    # 孤立: 与前无关, 与后无关
    for year, (old, new) in new_die_isolate_judge.items():
        new_born.setdefault(year, list()).extend(list(set(old) - set(new)))
        die_out.setdefault(year, list()).extend(list(set(new) - set(old)))
        isolate.setdefault(year, list()).extend(list(set(old) & set(new)))

    years = sorted(list(years))
    json_save(new_born, f"./data/comm_evo/{model_}/{filter_set}/new_born.json")
    json_save(die_out, f"./data/comm_evo/{model_}/{filter_set}/die_out.json")
    json_save(inherit, f"./data/comm_evo/{model_}/{filter_set}/inherit.json")
    json_save(divide, f"./data/comm_evo/{model_}/{filter_set}/divide.json")
    json_save(merge, f"./data/comm_evo/{model_}/{filter_set}/merge.json")
    json_save(isolate, f"./data/comm_evo/{model_}/{filter_set}/isolate.json")
    json_save(years, f"./data/comm_evo/{model_}/{filter_set}/years.json")
