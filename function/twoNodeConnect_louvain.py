import community as community_louvain
import copy
import networkx as nx
from networkx import k_edge_components as k_connect
from networkx.algorithms.community import label_propagation_communities as ISLPA
from function.common.slpa import slpa as SLPA
from networkx.algorithms.community import asyn_lpa_communities as LPA
from networkx.algorithms.community import girvan_newman as GN
from networkx.algorithms.community import greedy_modularity_communities as CNM
from networkx.algorithms.community import asyn_fluidc as FC
from networkx import k_core
import pickle
import csv
import json
import os
from collections import Counter
from function.connect.two_edge_node_connect import tarjan as two_connect
import math


def json_save(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def pickle_save(object_in, filename):
    f = open(filename, 'wb')
    pickle.dump(object_in, f)
    f.close()


def eliminate_comm(g_, use_node_):
    """
    :param g_: 图网络
    :param use_node_: 欲删除的点集合
    :return: 对于图g_, 删除use_node_点, 并返回剩余图的一边连通结果
    """
    g_copy = nx.Graph(g_)
    g_copy.remove_nodes_from(use_node_)
    one_connect = k_connect(g_copy, 1)
    eliminate_result = [list(val) for val in list(one_connect)]
    return eliminate_result


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


def func_choose(g_, func_name):
    if func_name == "louvain":
        lv_temp = community_louvain.best_partition(g_)
        dict_out = {}
        for key, value in lv_temp.items():
            dict_out.setdefault(value, []).append(key)
        result = [list(v) for v in dict_out.values()]
    elif func_name == "lpa":
        lpa_temp = LPA(g_, weight="weight")
        result = [list(val) for val in lpa_temp]
    elif func_name == "fluid":
        n = len(g_.nodes)
        k = math.floor(pow(1.7, math.log(n)))
        fc_temp = FC(g_, k)
        result = [list(val) for val in fc_temp]
    elif func_name == "islpa":
        islpa_temp = ISLPA(g_)
        result = [list(val) for val in islpa_temp]
    elif func_name == "slpa":
        slpa_temp = SLPA(g_, 30, 0.33)
        result = [list(v) for v in slpa_temp]
    elif func_name == "gn":
        gn_temp = GN(g_)
        result = [list(val) for val in next(gn_temp)]
    elif func_name == "cnm":
        cnm = CNM(g_)
        result = [list(val) for val in cnm]
    elif func_name == "k_core":
        k_core_result = k_core(g_)
        k_result = [list(k_core_result)]
        use_node = [node for comm in k_result for node in comm]
        eliminate_result = eliminate_comm(g_, use_node)
        result = k_result + eliminate_result
    else:
        return print("---请重新选择您的社区划分算法!---")
    return result


def twoConnect_louvain(g, describe="Karate_club", func_1="louvain", func_2="louvain"):
    """
        :param g: 图G
        :param describe: 二点连通社区划分结果，默认为Karate_club
        :param func_1: 选择最大社区的划分算法, 默认为louvain
        :param func_2: 选择最后聚点的划分算法，默认为louvain
        :return: lv_reduce: 列表，二次社区划分结果。
    """
    """
    举例(Facebook数据集)：
        method_new(g): 对Karate_club三边连通结果进行社区二次划分；
        method_new(g, "Facebook"): 对Facebook三边连通结果进行社区二次划分；
    """
    """
    ****************如需了解算法原理****************
    该算法思路，对二点连通结果的每个划分社区：如[[社区1], [社区2], [社区3]]中的社区1、2、3
    统计他们内部节点的邻居：如社区1有节点-node1、node2，与node1有连接边的点为-node2、node3、node4，与node2有连接边的点为-node1、node4
    则社区1的内部节点的邻居统计为-{社区1: {node1: 1, node2: 1, node3: 1, node4: 2}}，社区2、3同理
    统计每个节点属于的社区号：如节点node1、node2属于社区1，则{node1: 1, node2: 1}，其他节点同理
    由此，则很容易知道社区1的聚点自环边为(1+1)/2，社区1聚点和其他社区聚点的边的权重也可以通过邻居统计和社区号计算得出：
    如社区1存在node3的邻接点，即社区1与node3有边相连，则通过查node3所属的社区号，可得到社区1与node3所属社区的边的权重+1，以此类推
    ！！！！！！-------
    但上述操作仅适合对简单社区进行聚点操作，由于我们需要对二点连通结果的最大社区进行进一步划分，故上述操作需要进行一定的更改。
    我们依旧统计了内部节点邻居、节点所属社区号，对于自环边，依旧是按照上述操作进行。
    ------
    但是对于聚点与聚点的边，由于存在以下情况：假设社区1中的割点为node1，且我们需要对社区1进行二次划分（即社区1为最大社区），划分成[社区1_1，社区1_2]，
    割点node1分到了社区1_1，假设社区2同样存在割点node1，则对于社区2与社区1_1的权重显然存在，这是我们需要的。但是因为
    社区1_2的节点node2与割点node1相连，所以社区2的邻居包含了node2，即社区2与社区1_2的权重也会存在，但这不是我们需要的，并且它会作为干扰难以去除。
    ------
    所以对于聚点-聚点边，我们需要分类操作：
    1、我们将最大社区孤立，并按照前面的方式得到二次划分后聚点-聚点边的权重
    2、完成上述操作后，我们利用割点的原理--割点作为被划分社区的唯一连接点。可以很清楚的得到，任意两个社区A、B(含割点的社区)的聚点-聚点边的权重，
       就是邻居统计中社区A中割点出现的次数，社区B中割点出现的次数的总和。
    ------
    按照上述原理，得到下面的代码运算。
    PS：为了方便代码理解，建议认真了解每一步的产出内容
    """

    # 读取二点连通结果
    # Connect_result：按照社区大小排序的二点连通结果-为选取最大社区使用   [社区1[], 社区2[], 社区3[]]
    # Cut_node：二点连通的割点-为统计聚点-聚点边权重使用   [割点1, 割点2, 割点3]
    Cut_node, Cut_edge, Connect_result, Connect_result2 = two_connect(nx.to_dict_of_lists(g))
    Connect_result = sorted(Connect_result, key=lambda x: len(x), reverse=True)  # 社区 按大小排序

    # 对最大社区进行louvain算法 此处算法任意，lpa也可以，但是结果必须转化成 [[][][][]] 形式
    # BigOne_result：lv社区划分结果的复制品-方便更改对最大社区划分的算法   [社区1_1[], 社区1_2[]]
    g_subgraph = g.subgraph(Connect_result[0])

    big_one = func_choose(g_subgraph, func_1)

    BigOne_result = copy.deepcopy(big_one)

    # 将二次划分结果整合会回原始划分结果
    # Connect_result_BigOne_Decompose：二次划分整合结果-为接下来的邻居统计和节点社区号统计使用   [社区1_1[], 社区1_2[], 社区2[]]
    Connect_result_BigOne_Decompose = list()
    Connect_result_BigOne_Decompose.extend(BigOne_result)
    Connect_result_BigOne_Decompose.extend(Connect_result[1:])

    # 节点社区号统计、社区邻居统计
    # node_commNum: {节点: 社区号} 对社区划分结果的每个点进行编号，方便确定其属于哪个社区
    # commNum_neighborCount：{社区号: {节点: 该节点在该社区号内出现次数}}
    # commNum_neighborCount_copy：邻居统计复制品-因为在聚点自环边统计中，需要删除邻居统计中的部分点
    node_commNum = dict()
    commNum_neighbor = dict()
    commNum = -1
    for comm in BigOne_result:
        commNum += 1
        for node in comm:
            node_commNum.setdefault(node, []).append(commNum)  # node_commNum: {节点: 社区号}
            commNum_neighbor.setdefault(commNum, []).extend(g_subgraph.neighbors(node))  # 邻居收集
    for comm in Connect_result[1:]:
        commNum += 1
        for node in comm:
            node_commNum.setdefault(node, []).append(commNum)  # node_commNum: {节点: 社区号}
            commNum_neighbor.setdefault(commNum, []).extend(g.neighbors(node))  # 邻居收集

    commNum_neighborCount = {commNum: dict(Counter(neighbor)) for commNum, neighbor in commNum_neighbor.items()}  # 邻居统计
    commNum_neighborCount_copy = copy.deepcopy(commNum_neighborCount)

    # 聚点自环边权重统计
    # edge_add：收集了聚点自环权重边的集合-这是整个算法的最后结果，聚点相关的权重边集合
    # commNum_neighborCount_comm：聚点-聚点权重边(最大社区二次划分的)初始化
    # node_comm：聚点还原统计：为了最后的社区划分还原使用   [聚点1: 社区1[], 聚点2: 社区2[]]
    # Cut_node_trans：社区蕴含割点集：为了后面聚合时候，去除重复割点
    edge_add = []
    commNum_neighborCount_comm = dict()
    node_comm = {}  # {聚点: 原点集}
    Cut_node_trans = dict()
    for commNum, comm in enumerate(Connect_result_BigOne_Decompose):
        node_center = 0
        if len(comm) != 1:
            for node in comm:
                node_center += commNum_neighborCount_copy[commNum][node]  # 聚点自环边的权重统计
                del commNum_neighborCount_copy[commNum][node]  # 点删除，为后面聚点-聚点边做准备
                node_comm.setdefault("C" + str(commNum), []).append(node)  # 聚点对应原点集，方便以后还原
        else:
            node_comm.setdefault("C" + str(commNum), []).append(comm[0])
        Cut_node_trans.setdefault("C" + str(commNum), [])
        edge_add.append(("C" + str(commNum), "C" + str(commNum), node_center / 2))
        commNum_neighborCount_comm[commNum] = dict()
    # 聚点-聚点边权重统计(最大社区二次划分的)
    # edge_add：收集了聚点-聚点权重边(最大社区二次划分的)的集合-这是整个算法的最后结果，聚点相关的权重边集合
    for commNum, neighborCount in commNum_neighborCount_copy.items():
        if commNum >= len(BigOne_result):  # 识别二次划分社区
            break
        for node, weight in neighborCount.items():
            node_commNum_temp = node_commNum[node][0]
            commNum_neighborCount_comm[commNum].setdefault(node_commNum_temp, 0)
            commNum_neighborCount_comm[commNum][node_commNum_temp] += weight  # 聚点-聚点权重统计
    # 聚点-聚点权重边：相互为边，weight会加2次，但是在最后进入重建g_new时，(C0, C1)和(C1，C0)的权重不会重复添加，所以weight不需要/2
    for commNum, neighborCount in commNum_neighborCount_comm.items():
        for commNum_1, weight in neighborCount.items():
            edge_add.append(("C" + str(commNum), "C" + str(commNum_1), weight))

    # 聚点-聚点边权重统计(非最大社区二次划分的)
    # edge_add：收集了聚点-聚点权重边(非最大社区二次划分的)的集合-这是整个算法的最后结果，聚点相关的权重边集合
    # Cut_node_trans：社区割点集统计 {社区：割点集[]}
    # Cut_node_comm：割点社区点集统计 {割点：社区集[]}
    Cut_node_comm = dict()
    for cut_node in Cut_node:
        cutNode_comm = node_commNum[cut_node]  # 以空手道为例：割点0所属社区有0 3 4
        for commNum_1 in range(len(cutNode_comm)):
            for commNum_2 in range(commNum_1 + 1, len(cutNode_comm)):
                weight_1 = commNum_neighborCount[cutNode_comm[commNum_1]][cut_node]  # 第一个循环值为 社区0中割点0的权重
                weight_2 = commNum_neighborCount[cutNode_comm[commNum_2]][cut_node]  # 第一个循环值为 社区3中割点0的权重
                edge_add.append(
                    ("C" + str(cutNode_comm[commNum_1]), "C" + str(cutNode_comm[commNum_2]), weight_1 + weight_2))
        Cut_node_comm[cut_node] = cutNode_comm
        for comm in cutNode_comm:
            Cut_node_trans["C" + str(comm)].append(cut_node)

    # 最大社区的每个聚点社区都可能与总内含割点但非自身内含割点有联系，其他聚点社区只能通过内含割点与其他社区有联系
    # comm_cutNode_weight: 社区割点权重集{社区：{割点：社区与割点的边数}}
    count = 0
    comm_cutNode_weight = dict()
    cut_node_collect = list()
    for gather_node, cut_node_list in Cut_node_trans.items():
        if count < len(BigOne_result):
            count += 1
            cut_node_collect.extend(cut_node_list)
    count = -1
    for gather_node, cut_node_list in Cut_node_trans.items():
        count += 1
        if count < len(BigOne_result):
            for cut_node in cut_node_collect:
                if commNum_neighborCount[int(gather_node[1:])].get(cut_node):
                    comm_cutNode_weight.setdefault(
                        gather_node, dict()).setdefault(cut_node, commNum_neighborCount[int(gather_node[1:])][cut_node])
        else:
            for cut_node in cut_node_list:
                comm_cutNode_weight.setdefault(
                    gather_node, dict()).setdefault(cut_node, commNum_neighborCount[int(gather_node[1:])][cut_node])

    # 以edge_add建立新Graph图
    # g_new：聚点成品图
    g_new = nx.Graph()
    g_new.add_weighted_edges_from(edge_add)

    # 使用louvain算法再次社区划分 此处算法任意，lpa也可以，但是结果必须转化成 [[][][][]] 形式
    # lv：louvain算法社区划分结果
    end_result = func_choose(g_new, func_2)

    # 聚点还原，显示真正的社区划分结果
    # lv_reduce：聚点还原后的社区划分结果  [社区1[], 社区2[], 社区3[]]
    lv_reduce = list()

    # 新：多重割点可能被分离到不同的社区。为了维护非重叠，仅保留首次出现社区的割点
    # 故新写一个统计割点集：cut_node_add
    cut_node_add = set()
    for comm in end_result:
        lv_reduce_temp = list()
        Cut_node_temp = dict()
        for node in comm:
            lv_reduce_temp.extend(node_comm[node])
            for cut_node_temp in Cut_node_trans[node]:
                if cut_node_temp in cut_node_add:
                    Cut_node_temp.setdefault(cut_node_temp, 0)
                    Cut_node_temp[cut_node_temp] += 1
                else:
                    Cut_node_temp.setdefault(cut_node_temp, -1)
                    Cut_node_temp[cut_node_temp] += 1
        # 大循环内的第一个循环结束，表示该社区的割点数已经统计完成
        # 删除重复点-总是为割点
        for key, val in Cut_node_temp.items():
            cut_node_add.add(key)
            if val:
                for count in range(val):
                    lv_reduce_temp.remove(key)
        # 此处割点删除完成
        lv_reduce.append(lv_reduce_temp)

    if not os.path.exists("./save/connect/" + describe):  # 如果没有describe文件夹时创建文件夹
        os.makedirs("./save/connect/" + describe)

    print_csv("./save/connect/" + describe + "/" + "node.csv", Cut_node, 1)                  # 割点储存
    print_csv("./save/connect/" + describe + "/" + "connect.csv", Connect_result)         # 二点连通储存
    print_csv("./save/connect/" + describe + "/" + "bigOne.csv", BigOne_result)           # 最大社区二次划分储存
    print_csv("./save/connect/" + describe + "/" + "lv_gather.csv", end_result)                   # 聚点社区划分储存
    print_csv("./save/connect/" + describe + "/" + "endResult.csv", lv_reduce)            # 聚点还原最终社区划分储存
    pickle_save(g_new, "./save/connect/" + describe + "/" + "gNew.graph")                 # 聚点图储存
    json_save("./save/connect/" + describe + "/" + "nodeComm.json", Cut_node_comm)        # 割点社区集储存
    json_save("./save/connect/" + describe + "/" + "commNode.json", comm_cutNode_weight)  # 社区割点权重集储存
    json_save("./save/connect/" + describe + "/" + "gather_comm.json", node_comm)         # 聚点还原集储存

    return lv_reduce
