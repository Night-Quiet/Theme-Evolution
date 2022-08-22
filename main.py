from function.func_ import *
from function.draw_ import draw_word_evo


"""数据预处理--将所有数据放到一个csv文件中, 并按照出版时间排序"""
# 生成文件: paper.csv & paper_update.csv % paper_end.csv
# data_preprocess()


"""关键词处理--将关键词 & 摘要词性还原, 过滤停用词"""
# 使用文件: paper_end.csv & paper_stopword.json
# 生成文件: paper_word.csv
# key_preprocess("simi")


"""将时间使用时间序列方法分段"""
# 使用文件: paper_word.csv
# 生成文件: 分段线性时间划分图.svg, 分段线性时间划分图.html, key_time.json, key_time_len.json
# get_time_line(limit_=5, remove_big=False, combine=0)


"""将关键词和摘要都按照分段年份聚集"""
# 使用文件: paper_word.csv, key_time.json
# 生成文件: sentence_gather.json, abstract_gather.json
# time_gather()


"""根据分段年份分别建立图G"""
# 使用文件: sentence_gather.json, key_time_len.json
# 生成文件: graph图文件 & model词模型文件
# build_graph(model_="word2vec", thresh=0.3)
# build_graph(model_="node2vec")
# build_graph(model_="jaccard")


"""将图G社区划分, 并获取社区核心词"""
# 使用文件: graph图文件
# 生成文件: subject_words.json, mix_subject_words.json
model = ["word2vec", "node2vec", "jaccard"]
filter_set = ["degree", "z_score"]
for fs in filter_set:
    for m in model:
        handle_g(model=m, filter_set=fs)


"""计算相邻年段的社区相关"""
# 使用文件: subject_words.json & model词模型文件
model = ["word2vec", "node2vec", "jaccard"]
filter_set = ["degree", "z_score"]
for fs in filter_set:
    for m in model:
        calc_relevance(model_=m, filter_set=fs)


"""时间关键词全词演绎图"""
model = ["word2vec", "node2vec", "jaccard"]
filter_set = ["degree", "z_score"]
for fs in filter_set:
    for m in model:
        comm_evo(model_=m, filter_set=fs)
        draw_word_evo(model_=m, filter_set=fs)





