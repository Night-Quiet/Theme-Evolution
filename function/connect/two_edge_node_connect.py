from collections import namedtuple


"""
使用----from twoConnectAll import tarjan as tarjan_two_all
为了阻止疑惑的发生，只需要看函数tarjan(g)的except StopIteration下面的return
就能了解什么变量储存了什么内容，理论上这个文件的代码不能改变
如果有需要可以直接联系老师，找到这个代码的作者
"""


TarjanContext = namedtuple('TarjanContext',
                           ['g',  # the graph                            # 邻接表
                            'S',  # The main stack of the alg.           # 储存已访问节点
                            'S_set',  # == set(S) for performance        # 储存已访问节点(集合形式)
                            'index',  # { v : <index of v> }             # 节点值: 节点下标
                            'lowlink',  # { v : <lowlink of v> }         # 节点值: 最小时间戳
                            'T',  # stack to replace recursion           # tarjan深度优先路径储存
                            'ret',                                       # 储存边双连通分量
                            'father'])  # return code                    # 节点的父节点

child = 0
path_road = []
cutNodeJudge = []
cutEdgeJudge = []


def _tarjan_head(ctx, v, fa):
    ctx.index[v] = len(ctx.index)                                         # 节点下标
    ctx.lowlink[v] = ctx.index[v]                                         # low v: 最小时间戳
    ctx.S.append(v)                                                       # 节点值 访问顺序栈
    ctx.S_set.add(v)                                                      # 节点值 不重复
    it = iter(ctx.g.get(v, ()))                                           # 如果g无节点v 返回()
    ctx.T.append((it, False, v, None))                                    # 节点v的邻接表 False 节点v None
    ctx.father[v] = fa                                                    # 节点v的父节点


def _tarjan_body(ctx, it, v, root):
    global child, path_road, cutNodeJudge
    for w in it:                                                  # 记住it是迭代式分量 看代码第37行, 即遍历过的点不会再遍历, 测试过即可明白
        if w not in ctx.index:
            ctx.T.append((it, True, v, w))                        # 深度优先回溯 记录
            _tarjan_head(ctx, w, v)                               # 深度优先前进 记录
            path_road.append([v, w])                              # 深度优先 前进边 吞并
            if v == root:                                         # 只有root向前获取的才是child 返祖获取的不是child
                child += 1
            return
        if w in ctx.S_set:
            if w != ctx.father[v]:                                # 父边不走
                if ctx.lowlink[v] != min(ctx.lowlink[v], ctx.index[w]):
                    path_road.append([v, w])                      # 深度优先记录 返祖边 吞并
                ctx.lowlink[v] = min(ctx.lowlink[v], ctx.index[w])
    if ctx.lowlink[v] == ctx.index[v]:                            # 每个点所有邻接判断结束, 且到达割点处
        scc = []
        w = None
        while v != w:                                             # 直到v==w, 即直到S 吐 到了割点
            w = ctx.S.pop()                                       # S是深度优先按点 吞吐
            scc.append(w)
            ctx.S_set.remove(w)
        ctx.ret.append(scc)


def tarjan(g):
    global child, path_road, cutNodeJudge
    cutNodeJudge = []
    ctx = TarjanContext(
        g=g,
        S=[],
        S_set=set(),
        index={},
        lowlink={},
        T=[],
        ret=[],
        father={})
    cutNode = []
    cutEdge = []
    main_iter = iter(g)
    judge = True
    root = -1
    while True:
        try:
            v = next(main_iter)                                       # 得到字典key
        except StopIteration:
            return list(set(cutNode)), cutEdge, cutNodeJudge, ctx.ret  # 分别为割点、割边、二点连通、二边连通
        if v not in ctx.index:                                        # 不在节点储存处
            _tarjan_head(ctx, v, 0)
        if judge:
            root = v
            judge = False
        while ctx.T:
            it, inside, v, w = ctx.T.pop()
            if inside:                                                # 回溯开始
                ctx.lowlink[v] = min(ctx.lowlink[w], ctx.lowlink[v])

                # 割点 割边 点双连通分量
                if ctx.lowlink[w] >= ctx.index[v]:                    # 到达割点判断条件
                    cutNode.append(v)                                 # 割点记录
                    cut_temp = []                                     # 点双连通分量 开始获取
                    while True:
                        if len(path_road) > 0:
                            temp = path_road.pop()                    # 出栈 吐出
                        else:
                            break
                        for value in temp:
                            cut_temp.append(value)                    # 点双连通分量 吃每条边的点
                        if temp == [v, w]:                            # 深度优先到割点处 停止获取
                            break
                    cutNodeJudge.append(list(set(cut_temp)))
                if ctx.lowlink[w] > ctx.index[v]:                     # 割边判断 建议自己查看两个环以割点连接 两个环以割边连接的区别,方便理解
                    cutEdge.append([v, w])
            _tarjan_body(ctx, it, v, root)
        if len(path_road) > 0:                                        # 最终回到原点的最后剩余边
            temp_1 = []
            for value in path_road.pop():
                temp_1.append(value)
            cutNodeJudge.append(temp_1)
        if v == root and child < 2:                                   # 起始点判断 记住, 是按深度优先 以该点出发才算一个child, 回溯的点不算其child
            cutNode.remove(v)
