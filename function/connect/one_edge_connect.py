from collections import namedtuple


"""
使用----from oneConnect import tarjan as tarjan_one
为了阻止疑惑的发生，只需要看函数tarjan(g)的except StopIteration下面的return
就能了解什么变量储存了什么内容，理论上这个文件的代码不能改变
如果有需要可以直接联系老师，找到这个代码的作者
"""


TarjanContext = namedtuple('TarjanContext',
                           ['g',  # the graph                         # 邻接表
                            'S',  # The main stack of the alg.        # 储存已访问节点
                            'S_set',  # == set(S) for performance     # 储存已访问节点(集合形式)
                            'index',  # { v : <index of v> }          # 节点值: 节点下标
                            'lowlink',  # { v : <lowlink of v> }      # 节点值: 最小时间戳
                            'T',  # stack to replace recursion        # tarjan深度优先路径储存
                            'ret'])  # return code                    # 储存连通分量


def _tarjan_head(ctx, v):
    ctx.index[v] = len(ctx.index)                                      # 节点下标
    ctx.lowlink[v] = ctx.index[v]                                      # low v: 最小时间戳
    ctx.S.append(v)                                                    # 节点值 访问顺序栈
    ctx.S_set.add(v)                                                   # 节点值 不重复
    it = iter(ctx.g.get(v, ()))                                        # 如果g无节点v 返回()
    ctx.T.append((it, False, v, None))                                 # 节点v的邻接表 False 节点v None


def _tarjan_body(ctx, it, v):
    for w in it:                                           # 记住it是迭代式分量, 即遍历过的点不会再遍历, 测试过即可明白, 具体在代码26行
        if w not in ctx.index:
            ctx.T.append((it, True, v, w))                 # 深度优先回溯 记录
            _tarjan_head(ctx, w)                           # 深度优先前进 记录
            return
        if w in ctx.S_set:                                 # 碰到返祖边(点), 连通, 更新最小时间戳
            ctx.lowlink[v] = min(ctx.lowlink[v], ctx.index[w])
    if ctx.lowlink[v] == ctx.index[v]:                     # 回溯至当初碰到的返祖边(点), 获得结果
        scc = []
        w = None
        while v != w:                                      # 将回溯过程所有点吐出, 直到v==w(返祖点)
            w = ctx.S.pop()
            scc.append(w)
            ctx.S_set.remove(w)
        ctx.ret.append(scc)                                # 获取连通分量


def tarjan(g):
    ctx = TarjanContext(
        g=g,
        S=[],
        S_set=set(),
        index={},
        lowlink={},
        T=[],
        ret=[])
    main_iter = iter(g)
    while True:
        try:
            v = next(main_iter)                                # 得到字典key
        except StopIteration:
            return ctx.ret
        if v not in ctx.index:                                 # 不在节点储存处, 即开始, 或一个新的环
            _tarjan_head(ctx, v)
        while ctx.T:
            it, inside, v, w = ctx.T.pop()
            if inside:                                         # 回溯开始
                ctx.lowlink[v] = min(ctx.lowlink[w],
                                     ctx.lowlink[v])           # 回溯更新
            _tarjan_body(ctx, it, v)
