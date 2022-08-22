import copy
import plotly.figure_factory as ff
from function.func_no_care import json_load, find_year, json_save, z_score, pickle_load
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.express.colors import qualitative as cq
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx


# 分段线性时间划分
def draw_time_line(key_time, time_keyword_num):
    time_ = list(time_keyword_num.keys())
    key_num = list(time_keyword_num.values())

    df = pd.DataFrame(columns=["Time", "val", "group"])
    for ind, t_ in enumerate(time_):  # 此处获取的是总的: 时间-关键词数量
        df = df.append({"Time": str(t_), "val": int(key_num[ind]), "group": "Original curve"}, ignore_index=True)
    for ind in key_time:  # 此处获取的是转折点的: 时间-关键词数量
        df = df.append({"Time": str(time_[ind]), "val": int(key_num[ind]), "group": "Segmented curve"}, ignore_index=True)
    fig = px.line(data_frame=df, x="Time", y="val", line_group="group", color="group",
                  facet_row_spacing=0.01, facet_col_spacing=0.01, orientation='v',
                  color_discrete_sequence=cq.Alphabet, line_shape="linear", render_mode='auto',
                  labels={"val": "Number Of Keywords", "group": ""},
                  title="Piecewise Linear Time Division Diagram", height=700)
    fig.show()
    fig.write_image("./data/get_time_line/分段线性时间划分图.svg")
    fig.write_image("./data/get_time_line/分段线性时间划分图.jpg")
    fig.write_html(file="./data/get_time_line/分段线性时间划分图.html", default_width="100%", default_height="100%")


# word2vec二次采样前后的句子长度对比
def draw_sampled():
    """word2vec二次采样前后的句子长度对比"""
    before = json_load("./extra/skip_gram/二次采样前的句子长度情况.json")
    after = json_load("./extra/skip_gram/二次采样后的句子长度情况.json")
    df = pd.DataFrame(columns=["x", "y1", "y2"])
    num = 40
    for ind, (val1, val2) in enumerate(zip(before, after)):
        if ind == num:
            break
        df = df.append({"x": ind, "y1": int(val1), "y2": "采样前句子长度情况"}, ignore_index=True)
        df = df.append({"x": ind, "y1": int(val2), "y2": "采样后句子长度情况"}, ignore_index=True)
    fig = px.bar(data_frame=df, x="x", y="y1", facet_col_wrap=3, facet_row_spacing=0.01, color="y2",
                 facet_col_spacing=0.01, color_discrete_sequence=cq.Alphabet, orientation="v",
                 barmode="overlay", text_auto=".1f", title="二次采样句子长度前后对比图",
                 labels={"y1": "句子情况", "y2": "句子长度对比", "x": "句子"},
                 template=None, width=None, height=None)
    fig.show()
    fig.write_image(file="./extra/extra/二次采样词频前后对比图.svg", validate=True, engine="auto")
    fig.write_html(file="./extra/extra/二次采样词频前后对比图.html", default_width="100%", default_height="100%")


# 年份论文-关键词数量对比图
def draw_papernum_contrast():
    """画年份总论文-论文-关键词数量图"""
    '''收录期刊'''
    period = ["SCIENTOMETRICS", "RESEARCH EVALUATION", "LIBRARY & INFORMATION SCIENCE RESEARCH",
              "JOURNAL OF THE ASSOCIATION FOR INFORMATION SCIENCE AND TECHNOLOGY",
              "JOURNAL OF THE AMERICAN SOCIETY FOR INFORMATION SCIENCE AND TECHNOLOGY",
              "JOURNAL OF INFORMETRICS", "JOURNAL OF INFORMATION SCIENCE",
              "JOURNAL OF DOCUMENTATION", "INFORMATION PROCESSING & MANAGEMENT"]
    data = pd.read_csv("./data/key_preprocess/paper_word.csv", skip_blank_lines=True)
    # 按月论文数
    # 按月关键词数
    month_paper = dict()
    month_keyword = dict()
    for i in range(len(data)):
        time = data.loc[i, "Publication Year"]
        keyword = str(data.loc[i, "keyword_plus"])
        month_paper.setdefault(time, 0)
        month_keyword.setdefault(time, list())
        month_paper[time] += 1
        month_keyword[time].extend(keyword.split(";"))
    # 按年论文数
    poly_num = 0
    year_paper = dict()
    for key, value in month_paper.items():
        if poly_num == 0:
            poly_num += 1
            key_save = str(key)[:4]
            year_paper.setdefault(key_save, 0)
        else:
            poly_num += 1
            poly_num = np.divmod(poly_num, 12)[1]
        year_paper[key_save] += value
    # 按年关键词数
    poly_num = 0
    year_keyword = dict()
    for key, value in month_keyword.items():
        if poly_num == 0:
            poly_num += 1
            key_save = str(key)[:4]
            year_keyword.setdefault(key_save, list())
        else:
            poly_num += 1
            poly_num = np.divmod(poly_num, 12)[1]
        year_keyword[key_save].extend(value)
    year_keyword = {key: len(set(value)) for key, value in year_keyword.items()}
    year_paper.pop("2009")  # 删除2009影响
    year_keyword.pop("2009")

    # wos期刊论文总发行数
    period_num = json_load("./extra/extra/period_num.json")
    year_paper_all = dict()
    for per_, value in period_num.items():
        for year, num in value.items():
            year_paper_all.setdefault(year, 0)
            year_paper_all[year] += num

    def normal(dict_: dict):
        dict_return = dict()
        value_list = list(dict_.values())
        mean_ = np.mean(value_list)
        std = np.std(value_list)
        for key_, value_ in dict_.items():
            dict_return[key_] = (value_ - mean_) / std
        return dict_return

    year_paper_all, year_paper, year_keyword = normal(year_paper_all), normal(year_paper), normal(year_keyword)
    df = pd.DataFrame(columns=["x", "y", "y1", "g", "b"])
    for (year1, value1), (year2, value2), (year3, value3) in zip(year_paper_all.items(), year_paper.items(),
                                                                 year_keyword.items()):
        if year1 == "2022":
            break
        df = df.append({"x": year1, "y": value1,
                        "g": "Total Number Of Papers \nIn WOS Related Journals Each Year",
                        "Type": "All Field"}, ignore_index=True)
        df = df.append({"x": year2, "y": value2,
                        "g": "The Total Number Of Papers \nIn The Field In WOS \nRelated Journals Each Year",
                        "Type": "One Field"}, ignore_index=True)
        df = df.append({"x": year3, "y": value3,
                        "g": "The Number Of Keyword Types \nOf Papers In The Field \nIn WOS Related Journals Every Year",
                        "Type": "One Field"}, ignore_index=True)
    fig = px.line(data_frame=df, x="x", y="y", line_group="g", color="g", facet_row="Type",
                  facet_row_spacing=0.01, facet_col_spacing=0.01, orientation='v',
                  labels={"x": "Year", "y": "Quantity After Normalization", "g": ""},
                  color_discrete_sequence=cq.Alphabet, line_shape="spline", render_mode='auto',
                  template=None, width=None, height=700)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.show()
    fig.write_image(file="./extra/extra/论文-关键词种类数对比图.jpg", validate=True, engine="auto")
    fig.write_image(file="./extra/extra/论文-关键词种类数对比图.svg", validate=True, engine="auto")
    fig.write_html(file="./extra/extra/论文-关键词种类数对比图.html", default_width="100%", default_height="100%")


# 年份关键词前20最大词频分布图
def draw_keyword_num(num_max=20):
    data_ = pd.read_csv("./data/key_preprocess/paper_word.csv", skip_blank_lines=True)
    time_line = json_load("./data/get_time_line/key_time.json")
    year_keyword = dict()
    for i in range(len(data_)):
        # time_ = str(data_.loc[i, "Publication Year"])[:4]
        time_ = find_year(data_.loc[i, "Publication Year"], time_line)
        keyword = str(data_.loc[i, "keyword_plus"])
        year_keyword.setdefault(time_, list())
        year_keyword[time_].extend(keyword.split(";"))
    year_keyword.pop("200902")  # 删掉200902年的数据
    # year_keyword.pop("2022")
    year_keyword_pre = dict()
    for key, value in year_keyword.items():
        word_pre = dict(Counter(value))
        word_pre = sorted(word_pre.items(), key=lambda x: x[1], reverse=True)[:num_max]
        year_keyword_pre[key] = word_pre
    json_save(year_keyword_pre, "./extra/extra/time_word_pre.json")
    df = pd.DataFrame(columns=["Year", "Word", "pre"])
    for year, value in year_keyword_pre.items():
        for word, pre in value:
            df = df.append({"Year": year, "Word": word, "pre": pre}, ignore_index=True)
    fig = px.bar(data_frame=df, x="Word", y="pre", facet_col_wrap=3, color="Year",
                 facet_col_spacing=0.01, color_discrete_sequence=cq.Alphabet, orientation="v",
                 labels={"pre": "Word Frequency"},
                 barmode="relative", text_auto=".1f",
                 template=None, width=None, height=None)
    fig.show()
    fig.write_image(file="./extra/extra/年份关键词前20最大词频分布图.jpg", validate=True, engine="auto", height="600", width="700")
    fig.write_image(file="./extra/extra/年份关键词前20最大词频分布图.svg", validate=True, engine="auto", height="600", width="700")
    fig.write_html(file="./extra/extra/年份关键词前20最大词频分布图.html", default_width="100%", default_height="100%")


# 时间关键词演绎图
def draw_time_evo(model, is_show=False):
    relevance = json_load("./data/calc_relevance/"+model+"_relevance.json")
    nodes = list()
    edges = list()
    colors = dict()
    num = 0
    for ind, (key, value) in enumerate(relevance.items()):
        year0, year1 = key.split("-")
        year0, year1 = year0[4:] + ": ", year1[4:] + ": "
        for ind2, (k, val) in enumerate(value.items()):
            for node1, node2, sims in val:
                node1 = year0 + str(node1)
                node2 = year1 + str(node2)
                nodes.append((node1, {"label": ind}))
                nodes.append((node2, {"label": ind + 1}))
                edges.append((node1, node2, sims))
                colors[node1] = num
                colors[node2] = num
            if val:
                num += 1
    g_draw = nx.Graph()
    g_draw.add_nodes_from(nodes)
    g_draw.add_weighted_edges_from(edges)
    pos = nx.multipartite_layout(g_draw, subset_key="label")
    pos_ = dict()
    for key, value in pos.items():
        pos_[key] = np.array([value[0] * 150, value[1] * 40])
    plt.figure(figsize=(150, 40))
    nx.draw(g_draw, pos=pos_, nodelist=list(colors.keys()), edgelist=list(g_draw.edges(data=True)),
            node_color=list(colors.values()),
            cmap=plt.cm.Pastel1, with_labels=True)
    plt.axis("equal")
    plt.savefig("./data/draw_time_evo/" + model + "时间关键词演绎图.pdf")
    plt.savefig("./data/draw_time_evo/" + model + "时间关键词演绎图.svg")
    if is_show:
        plt.show()


# 时间关键词全词演绎图
def draw_word_evo(model_, filter_set, is_show=False):
    new_born = json_load(f"./data/comm_evo/{model_}/{filter_set}/new_born.json")
    die_out = json_load(f"./data/comm_evo/{model_}/{filter_set}/die_out.json")
    inherit = json_load(f"./data/comm_evo/{model_}/{filter_set}/inherit.json")
    divide = json_load(f"./data/comm_evo/{model_}/{filter_set}/divide.json")
    merge = json_load(f"./data/comm_evo/{model_}/{filter_set}/merge.json")
    isolate = json_load(f"./data/comm_evo/{model_}/{filter_set}/isolate.json")
    mix_comm_link = json_load(f"./data/calc_relevance/{model_}/{filter_set}/mix_comm_link.json")
    years = [str(val) for val in json_load(f"./data/comm_evo/{model_}/{filter_set}/years.json")]
    g_no_word = pickle_load(f"./data/calc_relevance/{model_}/{filter_set}/g_no_word.graph")

    nodes = list()
    edges = list()
    colors = dict()
    evo_node = list()
    all_node = list()
    for ind, year in enumerate(years):
        new_born0 = new_born.get(year, list())
        die_out0 = die_out.get(year, list())
        inherit0 = inherit.get(year, list())
        divide0 = divide.get(year, list())
        merge0 = merge.get(year, list())
        isolate0 = isolate.get(year, list())
        for node in new_born0:
            nodes.append((node, {"group": ind}))
            colors[node] = 1
            evo_node.append(node)
        for node in die_out0:
            nodes.append((node, {"group": ind}))
            colors[node] = 2
            evo_node.append(node)
        for node in inherit0:
            nodes.append((node, {"group": ind}))
            colors[node] = 3
            evo_node.append(node)
        for node in divide0:
            nodes.append((node, {"group": ind}))
            colors[node] = 4
            evo_node.append(node)
        for node in merge0:
            nodes.append((node, {"group": ind}))
            colors[node] = 5
            evo_node.append(node)
        for node in isolate0:
            nodes.append((node, {"group": ind}))
            colors[node] = 6
            evo_node.append(node)

    for mix, (old_comm, new_comm) in mix_comm_link.items():
        nodes.append((mix, {"group": float(mix.split("-")[1])}))
        colors[mix] = 7
        for comm in old_comm:
            edges.append((mix, comm))
            all_node.append(comm)
        for comm in new_comm:
            edges.append((mix, comm))
            all_node.append(comm)
    for node in set(all_node) - set(evo_node):
        nodes.append((node, {"group": int(node.split("-")[1])}))
        colors[node] = 8

    # g_draw = nx.Graph()
    g_no_word.add_nodes_from(nodes)
    g_no_word.add_edges_from(edges)
    edge_size = {i[0:2]: i[2]['weight'] for i in g_no_word.edges(data=True)}
    nodes_size = dict()
    for i in g_no_word.nodes(data=True):
        if i[0][:3] == "mix":
            nodes_size[i[0]] = i[1]["score"] * 20
        else:
            nodes_size[i[0]] = i[1]["score"] * 200
    pos = nx.multipartite_layout(g_no_word, subset_key="group")
    pos_ = dict()
    for key, value in pos.items():
        pos_[key] = np.array([value[0] * 300, value[1] * 100])
    plt.figure(figsize=(300, 100))
    nx.draw(g_no_word, pos=pos_, nodelist=list(colors.keys()), edgelist=list(g_no_word.edges(data=True)),
            node_color=list(colors.values()), node_size=[nodes_size[node] for node in colors.keys()],
            cmap=plt.cm.Set1, with_labels=True, width=list(edge_size.values()), font_size=24)
    plt.axis("equal")
    plt.savefig(f"./data/draw_word_evo/{model_}/{filter_set}/时间关键词全词演绎图.pdf")
    plt.savefig(f"./data/draw_word_evo/{model_}/{filter_set}/时间关键词全词演绎图.svg")
    plt.savefig(f"./data/draw_word_evo/{model_}/{filter_set}/时间关键词全词演绎图.jpg")
    if is_show:
        plt.show()
    print("******** draw_word_evo() 完成 ********\n")


# 四种中心性+z得分的相似度表格
def draw_table(g):
    def cos_similar(x, y):
        """
        :param x: 向量x
        :param y: 向量y
        :return: 两者的余弦相似度
        """
        x = np.array(x)
        y = np.array(y)
        cos = np.dot(x, y) / np.sqrt(np.sum(np.square(x)) * np.sum(np.square(y)))
        return np.around(cos, 2)
    label = ['degree', 'betweenness', 'eigenvector', 'closeness', "z-score"]

    degree_ = nx.get_node_attributes(g, label[0])
    betweenness_ = nx.get_node_attributes(g, label[1])
    eigenvector_ = nx.get_node_attributes(g, label[2])
    closeness_ = nx.get_node_attributes(g, label[3])
    z_score_ = z_score(g, is_norm=True)

    label_value = [degree_, betweenness_, eigenvector_, closeness_, z_score_]
    centrality = [[], [], [], [], []]
    df2 = pd.DataFrame(columns=["label", "value"])
    for node in g.nodes():
        for i in range(len(label)):
            centrality[i].append(label_value[i][node])
            df2 = df2.append({"label": label[i], "value": label_value[i][node]}, ignore_index=True)

    df = pd.DataFrame(columns=label, index=label)

    for i in range(len(label)):
        for j in range(len(label)):
            df.loc[label[j], label[i]] = cos_similar(centrality[i], centrality[j])
    fig = ff.create_table(table_text=df, index=True, index_title="",
                          annotation_offset=0.45, height_constant=20, hoverinfo="none")
    fig.write_image(file="./extra/extra/五种中心性的相似度表格.jpg", validate=True, engine="auto")
    fig.write_image(file="./extra/extra/五种中心性的相似度表格.svg", validate=True, engine="auto")
    fig.write_html(file="./extra/extra/五种中心性的相似度表格.html", default_width="100%", default_height="100%")

    fig = px.histogram(data_frame=df2, x="value", color="label", pattern_shape="label", facet_row=None,
                       facet_col="label",
                       facet_col_wrap=3, facet_row_spacing=0.03, facet_col_spacing=0.03,
                       animation_frame=None, animation_group=None,
                       labels={"value": ""},
                       color_discrete_sequence=cq.Alphabet, color_discrete_map=None,
                       pattern_shape_sequence=['/', "|", "x", '+', '.'],
                       pattern_shape_map=None,
                       opacity=None, orientation="v", barmode="relative", barnorm=None, histnorm=None,
                       log_x=False,
                       log_y=False, range_x=[0, 1], range_y=[0, 300], histfunc="count", cumulative=False, nbins=1000,
                       text_auto=".1f", template=None, width=None, height=None)
    fig.write_image(file="./extra/extra/中心性测量分布图.jpg", validate=True, engine="auto")
    fig.write_image(file="./extra/extra/中心性测量分布图.svg", validate=True, engine="auto")
    fig.write_html(file="./extra/extra/中心性测量分布图.html", default_width="100%", default_height="100%")

