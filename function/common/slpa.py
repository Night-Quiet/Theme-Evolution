from collections import defaultdict
import numpy as np


def slpa(g, t, r):
    """
    :param g: 图G
    :param t: 迭代次数
    :param r: 社区阈值，取值(0, 1)
    :return: result: 社区列表[[], []]
    """
    """网上抄的SLPA算法，并加以改进"""
    # 将图中数据录入到数据字典中以便使用
    weight = {j: {} for j in g.nodes()}
    for q in weight.keys():
        for m in g[q].keys():
            weight[q][m] = g[q][m]['weight']
    # 建立成员标签记录
    memory = {i: {i: 1} for i in g.nodes()}
    # 开始遍历T次所有节点
    for t_one in range(t):
        listenerslist = list(g.nodes())
        # 随机排列遍历顺序
        np.random.shuffle(listenerslist)
        # 开始遍历节点
        for listener in listenerslist:
            # 每个节点的key就是与他相连的节点标签名
            speakerlist = g[listener].keys()
            if len(speakerlist) == 0:
                continue
            labels = defaultdict(int)
            #
            for j, speaker in enumerate(speakerlist):
                total = float(sum(memory[speaker].values()))
                # 查看speaker中memory中出现概率最大的标签并记录，key是标签名，value是Listener与speaker之间的权
                labels[list(memory[speaker].keys())[
                    np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += \
                weight[listener][speaker]
            # 查看labels中值最大的标签，让其成为当前listener的一个记录
            maxlabel = max(labels, key=labels.get)
            if maxlabel in memory[listener]:
                memory[listener][maxlabel] += 1
            else:
                memory[listener][maxlabel] = 1.5
    # 提取出每个节点memory中记录标签出现最多的一个
    for primary in memory:
        total = float(sum(memory[primary].values()))
        p = list(memory[primary].keys())[
            np.random.multinomial(1, [freq / total for freq in memory[primary].values()]).argmax()]
        memory[primary] = {p: memory[primary][p]}
    # 如果希望将那种所属社区不明显的节点排除的就使用下面这段注释代码
    # '''
    for primary, change in memory.items():
        for change_name, change_number in set(change.items()):
            if change_number / float(t + 1) < r:
                del change[change_name]
    # '''
    communities = {}
    # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
    for primary, change in memory.items():
        for label in change.keys():
            if label in communities:
                communities[label].add(primary)
            else:
                communities[label] = set([primary])
    freecommunities = set()
    keys = list(communities.keys())
    # 排除相互包含的社区（上面那段注释代码不加这段也可以不加）
    # '''
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i + 1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                freecommunities.add(label0)
            elif comm0.issuperset(comm1):
                freecommunities.add(label1)
    for comm in freecommunities:
        del communities[comm]
    # '''
    # 理论上, 如果某个监听器的最大听众都不符合阈值, 那么它理论上是单人, 即单社群.
    result = list()
    list_node = set()
    g_node = set(g.nodes())
    for val in communities.values():
        result.append(val)
        list_node |= val
    surplus_node = g_node - list_node
    for val in surplus_node:
        result.append(set([val]))
    # 返回值是个社区列表
    return result