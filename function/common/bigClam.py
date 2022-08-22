from function.func_ import *
import numpy as np


def l(f_m, e_m):
    # 计算l(F)=logP(G|F)
    stats_sum = 0
    for row in range(len(e_m)):
        for column in range(len(e_m)):
            fu = f_m[row]
            fv = f_m[column]
            if e_m[row, column] == 1:
                stats_sum += np.log(1-np.exp(-np.dot(fu, fv)))
            else:
                stats_sum -= np.dot(fu, fv)
    return stats_sum


def l_grad(f_m, e_t, u, f_m_sum):
    # 计算某节点导数 | 梯度
    stats_sum = np.zeros(len(f_m))
    fu = f_m[u]
    for nei in e_t[u]:
        fv = f_m[nei]
        stats_sum += np.divide(fv * np.exp(-np.dot(fu, fv)), (1-np.exp(-np.dot(fu, fv))))
        stats_sum += fv
    stats_sum -= f_m_sum
    stats_sum += fu
    return stats_sum


def bigClam(g):
    dict_list = nx.to_dict_of_lists(g)

    gLen = len(g.nodes())
    node_index = {node: index for index, node in enumerate(g.nodes())}  # 先做节点和索引的映射表

    # 建立基于节点索引对应的邻接矩阵 邻接表
    edge_matrix = np.zeros([gLen, gLen], dtype=int)
    edge_table = dict()
    for key, values in dict_list.items():
        for value in values:
            edge_matrix[node_index[key], node_index[value]] = 1
            edge_table.setdefault(node_index[key], list()).append(node_index[value])

    f_matrix = np.random.rand(gLen, gLen)  # 建立归属强度矩阵F，并随机化--记住，范围[0, 1]

    fai = 0.005  # 学习速率
    for u in range(gLen):
        f_matrix_sum = np.sum(f_matrix, axis=0)
        ten = 0
        while True:
            l_g = l_grad(f_matrix, edge_table, u, f_matrix_sum)
            f_matrix[u] += fai*l_g
            # 元素 非负处理
            f_matrix[u] = np.maximum(0.001, f_matrix[u])
            ten += 1
            if ten == 1000:
                break

    # 开始算概率 为0概率设为：
    p_zero = np.divide(2*gLen, len(list(g.edges()))*(len(list(g.edges()))-1))
    p_matrix = np.zeros([gLen, gLen], dtype=float)
    p_network = dict()
    for row in range(gLen):
        for column in range(gLen):
            p = 1 - np.exp(float(-np.dot(f_matrix[row], f_matrix[column])))
            if p != 0:
                p_matrix[row, column] = p
            else:
                p_matrix[row, column] = p_zero

    # 每个节点相对最大概率进行归一化处理
    p_argMax = np.argmax(p_matrix, 1)
    for row in range(gLen):
        p_divide = p_argMax[row]
        for column in range(gLen):
            p_matrix[row, column] = np.divide(p_matrix[row, column], p_matrix[row, p_divide])
            if p_matrix[row, column] > 0.5:
                p_network.setdefault(column, list()).append(row)

    return p_network









