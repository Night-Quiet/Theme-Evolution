import time
from collections import namedtuple
import copy
import sys
from function.SingleLinkList import SingleLinkList as sl

"""
使用----from threeConnect import tarjan as tarjan_three
为了阻止疑惑的发生，只需要看函数-tarjan(g)-的-except StopIteration-下面的-return
就能了解什么变量储存了什么内容，理论上这个文件的代码不能改变
如果有需要可以直接联系老师，找到这个代码的作者

这里面的代码注释是最齐全的，配合论文能很容易理解三边连通的运行思路
并且记住，这里每个变量，每个操作都是有意义的，有价值的，任何没有把握的删改，都会导致整个算法彻底错误，请谨慎
"""


TarjanContext = namedtuple('TarjanContext',
                           ['g',  # the graph                         # 邻接表
                            'S',  # The main stack of the alg.        # 储存已访问节点
                            'S_set',  # == set(S) for performance     # 储存已访问节点(集合形式)
                            'index',  # { v : <index of v> }          # 节点值: 节点下标
                            'lowlink',  # { v : <lowlink of v> }      # 节点值: 最小时间戳
                            'T',  # stack to replace recursion        # tarjan深度优先路径储存
                            'ret',                                    # 此处未使用
                            'father',                                 # 节点的父节点
                            'path',                                   # 节点的挂边
                            'tree_end',                               # 节点子树的最终涉及范围
                            'deg',                                    # 节点的度
                            "zu",                                     # 节点吞并分组
                            ])  # return code


def key_value(dict_in):
    """
    字典的{key: value}转换为{value: key}, 然后获取新的value组成的列表
    """
    dict_out = {}
    for key, value in dict_in.items():
        dict_out.setdefault(value, []).append(key)
    return dict_out


def judge_edge(ctx, v, w):
    """
    判断节点w是否在节点v的子树范围内
    """
    if w is None or ctx.index[v] <= ctx.index[w] <= ctx.tree_end[v]:
        return False
    else:
        return True


def absorb(ctx, v, p, other=None):
    """
    吞吐机制
    """
    p_path = ctx.path[p]                                       # p节点挂边: p-path
    while True:
        if len(p_path) <= 0:                                   # 如果挂边长度缩减为0, 结束
            break
        node_temp = p_path[-1]                                 # 从后开始获取每个挂边点
        if other is not None:                                  # 如other!=None，进行的是前向边的吞吐
            if judge_edge(ctx, node_temp, other):              # <--，直到other不是node_temp的子树点停止，不是全部吞并
                break
        node_temp = p_path.pop()                               # p-path从后开始删除点
        if node_temp != v:                                     # node_temp为v时，吞并无意义，所以不为v时吞并
            ctx.zu[v].append_list(ctx.zu[node_temp].head(), ctx.zu[node_temp].end())
            ctx.zu[node_temp].empty()                          # node_temp节点被吞并，则其被清空，防止其他点对其二次吞并
            ctx.deg[v] += ctx.deg[node_temp] - 2               # v必定和node_temp有关联边，吞并后v与其相关的度被删减
            

def _tarjan_head(ctx, v, fa):
    """
    节点信息更新
    """
    ctx.index[v] = len(ctx.index)                              # 节点下标
    ctx.lowlink[v] = ctx.index[v]                              # low v: 最小时间戳
    ctx.S.append(v)                                            # 节点值 访问顺序栈
    ctx.S_set.add(v)                                           # 节点值 不重复
    it = iter(ctx.g.get(v, ()))                                # 如果g无节点v 返回()
    ctx.T.append((it, False, v, None))                         # 节点v的邻接表 False 节点v None
    ctx.father[v] = fa                                         # 节点v的父节点


def _tarjan_body(ctx, it, v):
    """
    深度优先前进的操作
    """
    for w in it:                                              # 记住it是迭代式分量 看代码第81行, 即遍历过的点不会再遍历, 测试过即可明白
        if w not in ctx.index:
            ctx.T.append((it, True, v, w))                    # 深度优先回溯 记录
            _tarjan_head(ctx, w, v)                           # 深度优先前进 记录
            return
        if w in ctx.S_set:
            if w != ctx.father[v]:                            # 父边不走
                if ctx.index[w] < ctx.lowlink[v]:             # 返祖边判断，可更新low，v吞并v-path，同时path增加自己恢复自身（没吐出点理应path算自己）
                    absorb(ctx, v, v)
                    ctx.path[v].append(v)
                elif ctx.index[w] > ctx.index[v]:             # 前向边判断，v吞并v-path(w--v部分)，同时path增加自己恢复自身
                    absorb(ctx, v, v, w)
                    ctx.path[v].append(v)
                    ctx.deg[v] -= 2                           # 前向边完成吞并，说明v与w有两条关联边，故v的度需多-2
                ctx.lowlink[v] = min(ctx.lowlink[v], ctx.index[w])
    ctx.tree_end[v] = len(ctx.index)                          # v子树深度优先遍历中，涉及点范围的终点


def tarjan(g):
    """
    深度优先开始以及回溯的操作
    """
    ctx = TarjanContext(
        g=g,
        S=[],
        S_set=set(),
        index={},
        lowlink={},
        T=[],
        ret=[],
        father={},                                            # 节点的父节点-初始设置为空
        path={key: [key] for key in g.keys()},                # 节点的携带路径-初始设置为自身
        deg={key: len(g[key]) for key in g.keys()},           # 节点度-初始设置为原图的度
        tree_end={key: sys.maxsize for key in g.keys()},      # 节点子树深度-初始设置为最大数
        zu={key: sl(key) for key in g.keys()},                # 节点分组-初始设置为以自身为标识的个人组
    )
    main_iter = iter(g)
    while True:
        try:
            v = next(main_iter)                               # 得到新的节点
        except StopIteration:                                 # 所有节点全部遍历完，返回
            need_result = [list(val.items()) for val in ctx.zu.values() if not val.is_empty()]
            return need_result
        if v not in ctx.index:                                # 此处为了新的连通分支，因为图很可能不是一个连通分支
            _tarjan_head(ctx, v, -1)
        while ctx.T:
            it, inside, v, w = ctx.T.pop()                    # 此处尤其关心inside：为True则正在回溯，为False则正在前进（深度优先）
            if inside:                                        # 回溯开始
                if ctx.deg[w] == 2:                           # 吐状态: 吐状态关键操作就是Path吐掉自身，以防止之后被其他人重新吞并
                    ctx.path[w] = [val for val in ctx.path[w] if val != w]
                if ctx.lowlink[w] < ctx.lowlink[v]:           # 树边: 状态1-可更新low， v吞并v-path，并用w-path+v代替，v在后面，为了path的pop操作
                    absorb(ctx, v, v)
                    ctx.path[v] = copy.deepcopy(ctx.path[w]) + [v]
                else:                                         # 树边: 状态2-不更新low，v吞并w-path，w-path清零
                    absorb(ctx, v, w)
                ctx.lowlink[v] = min(ctx.lowlink[w], ctx.lowlink[v])
            _tarjan_body(ctx, it, v)
