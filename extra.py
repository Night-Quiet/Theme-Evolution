import pickle

from function.draw_ import *
from function.func_no_care import *
from function.twoNodeConnect_louvain import twoConnect_louvain


"""年份论文-关键词数量对比图"""
# draw_papernum_contrast()


"""年份关键词前20最大词频分布图"""
# draw_keyword_num(20)


"""四种中心性+z得分的相似度表格"""
# g = pickle_load("./data/build_graph/node2vec/graph/all.graph")
# draw_table(g)


"""社区演绎图 关键词演绎图制作(小样本)"""
# graphml_comm_evo()
# graphml_word_evo()
