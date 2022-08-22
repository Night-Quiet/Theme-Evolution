from networkx.algorithms.community import k_clique_communities as kcc_one
import networkx as nx


def k_clique(g):
    """
    :param g: 图G
    :return: set(result_all): 社区列表
    """
    """写成3-clique-->2-clique-->1-clique形式的社区划分算法"""
    G = nx.Graph(g)
    result_all = []
    kcc_temp_1 = list(list(val) for val in kcc_one(G, 3))
    result_all.extend(kcc_temp_1)
    test1_clique = []
    for val in kcc_temp_1:
        test1_clique.extend(val)
    G.remove_nodes_from(list(set(test1_clique)))
    kcc_temp_2 = list(list(val) for val in kcc_one(G, 2))
    result_all.extend(kcc_temp_2)
    test2_clique = []
    for val in kcc_temp_2:
        test2_clique.extend(val)
    G.remove_nodes_from(list(set(test2_clique)))
    for val in G.nodes():
        result_all.append(val)
    return result_all
