'''
from function.draw_ import draw_time_line
from function.func_no_care import *
from gensim.models import Word2Vec
from function.twoNodeConnect_louvain import twoConnect_louvain as TC
import joblib


def get_point(data_line_):
    begin = data_line_[0]
    end = data_line_[-1]
    k = (end-begin) / (len(data_line_)-1)
    point_distance = list()
    for ind, val in enumerate(data_line_):
        point_distance.append(np.abs(val-k*ind-begin)/np.sqrt(1+k**2))
    max_k = topk(point_distance, 5)

    return max_k


def get_key_time(data_, list_, pre_index, limit_=3):
    # 这里有一个非常帅的操作, 利用列表在函数内使用完不会自动摧毁机制, 避免了使用全局变量
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
    key_time.extend([0, len(keyword_num) + 12 -1])
    key_time = sorted(key_time, reverse=False)

    draw_time_line(key_time, time_keyword_num)  # 画基于时间分段的线形图

    key_time = np.array(list(time_keyword_num.keys()))[key_time]
    key_time = key_time.tolist()  # 折线图的转折点
    json_save(key_time, "./data/get_time_line/key_time.json")
    json_save(len(key_time), "./data/get_time_line/key_time_len.json")
    print(f"--------关键转折时间为: {key_time}")
    print("******** get_time_line() 完成 ********\n")


def time_gather():
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


def build_model(time_, thresh=0.25, model_="word2vec"):
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

    if model_ == "word2vec":
        word2vec_(abstract_, keyword, thresh, time_)
        # model = Word2Vec(sentences=abstract_, vector_size=300, window=3, min_count=3, sample=1e-3)
        # g = nx.Graph()
        # # 建边
        # abstract_surplus = set(list(model.wv.key_to_index.keys()))
        # keyword = [val for val in keyword if val in abstract_surplus]
        # for ind, key_1 in tqdm(enumerate(keyword), total=len(keyword), desc="图g边生成", ncols=100):
        #     for key_2 in keyword[ind + 1:]:
        #         sims = model.wv.similarity(key_1, key_2)
        #         if sims >= thresh:
        #             g.add_edge(key_1, key_2, weight=sims)
        # model.save("./data/build_graph/word2vec/model/"+time_+".model")
        # pickle_save(g, "./data/build_graph/word2vec/graph/" + time_ + ".graph")

    else:
        g_list = list()
        for ab_ in tqdm(abstract_):
            g_abstract = text_graph(ab_)
            topics_sub_abstract = sub_graph(g_abstract, keyword)
            g_list.append(topics_sub_abstract)
        if model_ == "node2vec":
            final_g = final_graph(g_list, "./data/build_graph/node2vec/graph/" + time_ + ".graph", True)
            node2vec_(final_g, time_)
        elif model_ == "jaccard":
            final_graph(g_list, "./data/build_graph/jaccard/graph/" + time_ + ".graph", True)


def build_graph(model_="word2vec", thresh=0.25):
    key_time_len = int(json_load("./data/get_time_line/key_time_len.json")) - 1
    build_model("all", thresh, model_)
    for ind in range(key_time_len):
        time_ = "year" + str(ind)
        build_model(time_, thresh, model_)
    print("******** build_graph() 完成 ********\n")


def handle_g(model_):
    key_time_len = int(json_load("./data/get_time_line/key_time_len.json")) - 1
    subject_words = dict()
    for year_ in range(key_time_len):
        g = joblib.load("./data/build_graph/"+model_+"/graph/year" + str(year_) + ".graph")

        tc = TC(g, "temp"+str(year_))
        tc_g = dict()
        for ind, v in enumerate(tc):
            tc_g["comm" + str(ind)] = nx.Graph(g.subgraph(list(v)))
        # 主题确认
        key_max = dict()
        for ind, g_val in enumerate(list(tc_g.values())):
            key_max[ind] = sorted(degree_centrality(g_val).items(), key=lambda x: x[1], reverse=True)[:4]

        subject_words[year_] = key_max
    json_save(subject_words, "./data/handle_g/"+model_+"_subject_words.json")
    print("******** handle_g() 完成 ********\n")


def calc_relevance(model_, thresh=0, is_print=True):
    subject_words = json_load("./data/handle_g/"+model_+"_subject_words.json")
    if model_ == "word2vec":
        # model = Word2Vec.load("./data/build_graph/word2vec/model/all.model")
        model = torch.load("./data/build_graph/word2vec/model/all.model")
        W = model[0].weight.data
        token_to_ind = json_load("./data/build_graph/word2vec/ind/all.json")
    elif model_ == "node2vec":
        model = torch.load("./data/build_graph/node2vec/model/all.model")
        node_num = json_load("./data/build_graph/node2vec/node_num/all_node_num.json")
        vectors = model(torch.arange(len(node_num), device="cuda")).cpu().detach().numpy()
    elif model_ == "jaccard":
        g_ = pickle_load("./data/build_graph/jaccard/graph/all.graph")
    s_temp = dict()
    for ind, data_ in subject_words.items():
        s_temp.setdefault(ind, list())
        for val0_ind, val0 in enumerate(data_.values()):
            s_temp[ind].append(list())
            for val1 in val0:
                s_temp[ind][val0_ind].append(val1[0])
    relevance = dict()
    for ind, data in s_temp.items():
        if int(ind) >= len(s_temp) - 1:
            break
        year_evo = "year" + str(ind) + "-year" + str(int(ind) + 1)
        relevance.setdefault(year_evo, dict())
        for comm_ind, comm in enumerate(data):
            for comm_ind_, comm_ in enumerate(s_temp[str(int(ind) + 1)]):
                comm_evo = "comm" + str(comm_ind) + "-comm" + str(comm_ind_)
                relevance[year_evo].setdefault(comm_evo, list())
                for val1 in comm:
                    if model_ == "word2vec":
                        val1_vector = W[token_to_ind[val1]]
                        cos = torch.matmul(W, val1_vector) / (
                                torch.sum(W * W, dim=1) * torch.sum(val1_vector * val1_vector) + 1e-9).sqrt()
                        cos = cos.cpu().numpy()
                    for val2 in comm_:
                        if model_ == "word2vec":
                            # sims = model.wv.similarity(val1, val2)
                            sims = cos[token_to_ind[val2]]
                            if sims >= thresh:
                                relevance[year_evo][comm_evo].append((val1, val2, np.float64(sims)))
                        elif model_ == "node2vec":
                            sims = cos_sim(vectors[node_num[val1]], vectors[node_num[val2]])
                            if sims >= thresh:
                                relevance[year_evo][comm_evo].append((val1, val2, np.float64(sims)))
                        elif model_ == "jaccard":
                            short_len = nx.shortest_path_length(g_, source=val1, target=val2)
                            if short_len > 2:
                                sims = 0
                            elif short_len == 2:
                                val1_n = set([val for val in g_.neighbors(val1)])
                                val2_n = set([val for val in g_.neighbors(val2)])
                                sims = 0.5 * len(val1_n & val2_n) / len(val1_n | val2_n)
                            else:
                                val1_n = set([val for val in g_.neighbors(val1)])
                                val2_n = set([val for val in g_.neighbors(val2)])
                                sims = len(val1_n & val2_n) / len(val1_n | val2_n)
                            if sims >= thresh:
                                relevance[year_evo][comm_evo].append((val1, val2, np.float64(sims)))

    json_save(relevance, "./data/calc_relevance/"+model_+"_relevance.json")
    if is_print:
        for key, value in relevance.items():
            line = ""
            print(line + key)
            for key1, value1 in value.items():
                line = "    "
                print(line + key1)
                line = "        "
                for value2 in value1:
                    print(line + str(value2))
    print("******** calc_relevance() 完成 ********\n")
'''


