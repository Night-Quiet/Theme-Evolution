import sys
import os
from function.func_ import *
from function.connect.three_edge_connect import tarjan as tarjan_three
from function.connect.two_edge_node_connect import tarjan as tarjan_two_all
from function.connect.two_edge_connect import tarjan as tarjan_two
from function.connect.one_edge_connect import tarjan as tarjan_one
import networkx as nx

sys.setrecursionlimit(100000000)  # 增大递归深度，也算对以前递归代码对的留念，此处没任何用处


def my_all_connect(g, type_set=0, describe="tempSave"):
    """
    :param g: 图G
    :param type_set: 0: 正常1-2-3社区划分算法, 1: 在二连通中获取割点、割边等结果，并进行变量储存. 默认为0
    :param describe: 字符串，此为后面储存进行分配地址，建议使用数据集名字. 默认为"tempSave", 意为临时储存
    :return: cut_all: 列表，层次社区划分结果。如type_set为1：割点割边等结果被储存至csv
    """
    """
    举例(Facebook数据集)：
        my_all_connect(g): 单纯对Facebook数据集进行社区划分；
        my_all_connect(g, 1, "Facebook"): 对Facebook数据集进行社区划分并完成需求储存，储存至./save/connect/Facebook/文件夹中
    """
    
    dict_file = nx.to_dict_of_lists(g)  # 读取G为我们需要的邻接表
    # print("tarjan算法开始")
    """tarjan算法"""
    
    """一边连通"""
    
    two_dict_file = []  # two_dict_file的形态为[{第一个一边连通结果邻接表},{第二个一边连通结果邻接表}]
    
    """此处操作为将原始图进行一边连通，将一边连通的每个结果换成邻接表，存到two_dict_file里"""
    OneEdgeConnect = tarjan_one(dict_file)               # 一边连通算法，结果OneEdgeConnect为[[第一个分量], [第二个分量]] (分量仅为点的集合)
    for val in OneEdgeConnect:
        temp = nx.subgraph(g, val)                       # 每个结果分量读成局部图
        two_dict_file.append(nx.to_dict_of_lists(temp))  # 将每个局部图转换成邻接表存起来
    """一边连通后的二连通"""

    need_node_result = []         # 储存割点变量
    need_edge_result = []         # 储存割边变量
    need_nodeConnect_result = []  # 储存二点连通变量
    need_edgeConnect_result = []  # 储存二边连通变量
    
    three_dict_file = []
    # three_dict_file的形态为[[{first-first}, {first-second}], [{second-first}, {second-second}]]
    # [[{第一个一边连通的第一个二边连通的邻接表},{第一个一边连通的第二个二边连通的邻接表}], [{第二个一边连通的第一个二边连通的邻接表}]]
    three_dict_file_temp = []
    # three_dict_file_temp的形态为[{everyone-first}, {everyone-second}]
    
    """此处操作为对每个一边连通进行二边连通，将每次的二边连通的每个结果换成邻接表，存到three_dict_file_temp里，然后并一起并到three_dict_file里"""
    if type_set == 0:
        for one_value in two_dict_file:
            TwoEdgeConnect = tarjan_two(one_value)
            for Connect_value in TwoEdgeConnect:
                temp = nx.subgraph(g, Connect_value)  # 每个结果分量读成局部图
                three_dict_file_temp.append(nx.to_dict_of_lists(temp))  # 将每个局部图转换成邻接表存起来
            three_dict_file.append(three_dict_file_temp)  # 并到一起
            three_dict_file_temp = []
    elif type_set == 1:
        for one_value in two_dict_file:
            node, edge, TwoNodeConnect, TwoEdgeConnect = tarjan_two_all(one_value)
            # 二连通算法，结果node为[该二连通的割点集]；结果edge为[该二连通的割边集]；
            # TwoNodeConnect为[该二连通算法的二点连通集]；TwoEdgeConnect为[该二连通算法的二边连通集] (后两者结果类同于一边连通)
            
            need_node_result.extend(node)                   # 割点收集
            need_edge_result.extend(edge)                   # 割边收集
            need_nodeConnect_result.extend(TwoNodeConnect)  # 二点连通结果收集
            need_edgeConnect_result.extend(TwoEdgeConnect)  # 二边连通结果收集
            # for val in need_edgeConnect_result:
            #     print(val)
            # print(need_nodeConnect_result)

            for Connect_value in TwoEdgeConnect:
                temp = nx.subgraph(g, Connect_value)  # 每个结果分量读成局部图
                three_dict_file_temp.append(nx.to_dict_of_lists(temp))  # 将每个局部图转换成邻接表存起来
            three_dict_file.append(three_dict_file_temp)  # 并到一起
            three_dict_file_temp = []
    else:
        return print("请重新设置type_set参数!")

    """二边连通后三边连通"""

    cut_all = list()  # cut_all的形态为[[first-first-first社区], [first-first-second社区], [first-second-first社区]]
    
    """此处操作为将每个一边连通分量的每个二边连通分量进行三边连通划分，将结果并到cut_all里"""
    for oneValue in three_dict_file:
        for twoValue in oneValue:
            ThreeEdgeConnect = tarjan_three(twoValue)  # 三边连通算法，结果类同于一边连通
            cut_all.extend(ThreeEdgeConnect)           # 结果并到一起

    if type_set == 1:
        if not os.path.exists("./save/connect/" + describe):       # 如果没有describe文件夹时创建文件夹
            os.makedirs("./save/connect/" + describe)
        print_csv("./save/connect/" + describe + "/" + describe + "_node.csv", need_node_result, 1)             # 割点储存
        print_csv("./save/connect/" + describe + "/" + describe + "_edge.csv", need_edge_result)                # 割边储存
        print_csv("./save/connect/" + describe + "/" + describe + "_nodeConnect.csv", need_nodeConnect_result)  # 二点连通储存
        print_csv("./save/connect/" + describe + "/" + describe + "_edgeConnect.csv", need_edgeConnect_result)  # 二边连通储存
        print_csv("./save/connect/" + describe + "/" + describe + "_edgeConnect_three.csv", cut_all)            # 三边连通储存
    
    return cut_all

