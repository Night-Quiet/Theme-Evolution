"""
这个代码专门处理从百度文库接口抓包抓出来的论文文献常用词
使用抓包工具: Fiddler Classic
使用辅助工具: Json在线解析网站--https://www.devtool.com/json.html
"""
from function.func_ import *


# 文件: 中英文论文常用词.js
# 地址: https://wenku.baidu.com/view/6acdd479a26925c52cc5bf73.html
def stopword1():
    doc = json_load("./stopwords/中英文论文常用词.js")
    doc = doc["document.xml"][0]["c"][1]["c"]
    words = list()
    for val in doc:
        word = val["c"].strip()
        # 空词跳过
        if word == "":
            continue
        # 中文跳过, 判定方式: ASCII编码不超过127
        not_en = False
        for c in word:
            if ord(c) > 127:
                not_en = True
                break
        if not_en:
            continue
        words.append(word)

    # 开始根据词语特征分割出缩写
    word_filter = list()
    for word in words:
        word_split = word.split(" ")
        if len(word_split) <= 1:
            word_filter.append(word.lower())
        else:
            word_ = " ".join(word_split[:-1])
            abbr = word_split[-1]
            # or前针对缩写方式, or后针对这个狗单词--The Engineering Index Ei
            if (abbr[0] == word_[0] and len(abbr) <= 8) or \
                    (len(abbr) <= 2 and abbr[0] == word_split[1][0] and 65 <= ord(abbr[0]) <= 90):
                word_filter.append(word_.lower())
                word_filter.append(abbr.lower())
            else:
                word_filter.append(word.lower())
    # 处理错误词['medical literature analysis and', 'madlars', 'retrieval system']
    # 很显然, 中间的缩写对应的是前后两个词汇的组合, 应该是输入错误
    word_filter = list(set(word_filter))
    word_filter.remove("medical literature analysis and")
    word_filter.remove("retrieval system")
    word_filter.append("medical literature analysis and retrieval system")
    json_save(word_filter, "./stopwords/中英文论文常用词.json")


# 文件: 中英文论文常用词2.js
# 地址: https://wenku.baidu.com/view/1fc30ef2caaedd3383c4d3d3.html
def stopword2():
    doc = json_load("./stopwords/中英文论文常用词2.js")
    doc_concat = list()
    doc_concat.extend(doc["document.xml"][0]["c"][6:])
    doc_concat.extend(doc["document.xml"][1]["c"])
    doc_concat.extend(doc["document.xml"][2]["c"][:6])
    # 根据词语特征分割出缩写
    word_filter = list()
    for val in doc_concat:
        # 获取英文单词
        word = val["c"][0]["c"].strip()
        if word == "英文全称":  # 个别情况
            continue
        word_split = word.split(" ")
        if len(word_split) <= 1:
            word_filter.append(word.lower())
        else:
            word_ = " ".join(word_split[:-1])
            abbr = word_split[-1]
            # or前针对缩写方式, or后针对这个狗单词--The Engineering Index Ei
            if (abbr[0] == word_[0] and len(abbr) <= 8) or \
                    (len(abbr) <= 2 and abbr[0] == word_split[1][0] and 65 <= ord(abbr[0]) <= 90):
                word_filter.append(word_.lower())
                word_filter.append(abbr.lower())
            else:
                word_filter.append(word.lower())
    # 处理错误词['medical literature analysis and', 'madlars', 'retrieval system']
    # 很显然, 中间的缩写对应的是前后两个词汇的组合, 应该是输入错误
    word_filter = list(set(word_filter))
    word_filter.remove("medical literature analysis and")
    word_filter.remove("retrieval system")
    word_filter.append("medical literature analysis and retrieval system")
    json_save(word_filter, "./stopwords/中英文论文常用词2.json")


# 文件: 中英文论文常用词3.js
# 地址: https://wenku.baidu.com/view/80d50bbe3968011ca2009175.html
def stopword3():
    doc = json_load("./stopwords/中英文论文常用词3.js")
    doc_concat = list()
    doc_concat.append(doc["document.xml"][0]["c"][0]["c"][0]["c"])
    for ind in range(4):
        doc_concat.append(doc["document.xml"][0]["c"][ind+1]["c"][0]["c"][2:])
    # 根据词语特征分割出缩写
    word_filter = list()
    for val in doc_concat:
        # 获取英文单词
        val = val.replace(" ", " ")
        for word in val.split(","):
            word_filter.append(word.strip().lower())

    word_filter = list(set(word_filter))
    json_save(word_filter, "./stopwords/中英文论文常用词3.json")


# 文件: 论文中常用的词汇和短语.js
# 地址: https://wenku.baidu.com/tfview/3acf5a1c6137ee06eef9189f.html
def stopword4():
    doc = json_load("./stopwords/论文中常用的词汇和短语.js")
    doc = doc["body"][4:]
    words = list()
    for val in doc:
        word = val["c"].strip().strip(",")
        # 空词跳过
        if word == "":
            continue
        # 中文跳过, 判定方式: ASCII编码不超过127
        not_en = False
        for c in word:
            if ord(c) > 127:
                not_en = True
                break
        if not_en:
            continue
        words.extend(word.lower().split(","))

    word_filter = list(set(words))
    json_save(word_filter, "./stopwords/论文中常用的词汇和短语.json")


# 文件: 论文中常用的词汇和短语2.js
# 地址: https://wenku.baidu.com/tfview/3acf5a1c6137ee06eef9189f.html
def stopword5():
    doc = json_load("./stopwords/论文中常用的词汇和短语2.js")
    doc = doc["body"][3:]
    words = list()
    for val in doc:
        word = val["c"].strip().strip(",")
        # 空词跳过
        if word == "":
            continue
        # 中文跳过, 判定方式: ASCII编码不超过127
        not_en = False
        for c in word:
            if ord(c) > 127:
                not_en = True
                break
        if not_en:
            continue
        words.extend(word.lower().split(","))

    word_filter = list(set(words))
    json_save(word_filter, "./stopwords/论文中常用的词汇和短语2.json")


# 文件: 论文中常用的词汇和短语3
# 地址: https://wenku.baidu.com/view/41149b276f85ec3a87c24028915f804d2b16871e.html
def stopword6():
    doc = json_load("./stopwords/论文中常用的词汇和短语3.js")
    doc = doc["body"][1:]
    re_compile = re.compile(r"[,\)\(]+")
    words = list()
    for val in doc:
        word = val["c"].strip().strip(",")
        # 空词跳过
        if word == "":
            continue
        # 中文跳过, 判定方式: ASCII编码不超过127
        not_en = False
        for c in word:
            if ord(c) > 127:
                not_en = True
                break
        if not_en:
            continue
        words.extend(re.split(re_compile, word.lower()))

    word_filter = list(set(words))
    json_save(word_filter, "./stopwords/论文中常用的词汇和短语3.json")


# 文件: 论文中常用的词汇和短语4
# 地址: https://wenku.baidu.com/view/41149b276f85ec3a87c24028915f804d2b16871e.html
def stopword7():
    doc = json_load("./stopwords/论文中常用的词汇和短语4.js")
    doc = doc["body"][3:]
    re_compile = re.compile(r"[,\)\(]+")
    words = list()
    for val in doc:
        word = val["c"].strip().strip(",").strip("-")
        # 空词跳过
        if word == "":
            continue
        # 中文跳过, 判定方式: ASCII编码不超过127
        not_en = False
        for c in word:
            if ord(c) > 127:
                not_en = True
                break
        if not_en:
            continue
        words.extend(re.split(re_compile, word.lower()))

    word_filter = list(set(words))
    json_save(word_filter, "./stopwords/论文中常用的词汇和短语4.json")


# 文件: 慎用.txt
# 地址: https://www.kejixiezuo.com/article/189.html
def stopword8():
    word_filter = list()
    with open("./stopwords/慎用.txt", "r") as f:
        for line in f.readlines():
            word_filter.extend(line.strip("\n").split(" "))
    json_save(word_filter, "./stopwords/慎用.json")


# 文件: 慎用2.txt
# 地址:
# https://zhuanlan.zhihu.com/p/64564940
# https://zhuanlan.zhihu.com/p/74514116
# https://zhuanlan.zhihu.com/p/79654066
# https://zhuanlan.zhihu.com/p/77297252
# https://zhuanlan.zhihu.com/p/80384925
# https://zhuanlan.zhihu.com/p/81027816
def stopword9():
    word_filter = list()
    with open("./stopwords/慎用2.txt", "r") as f:
        for line in f.readlines():
            word_filter.extend(line.strip("\n").split(", "))
    json_save(word_filter, "./stopwords/慎用2.json")


# 全合并
sw1 = json_load("./stopwords/中英文论文常用词.json")
sw2 = json_load("./stopwords/中英文论文常用词2.json")
sw3 = json_load("./stopwords/中英文论文常用词3.json")
sw4 = json_load("./stopwords/论文中常用的词汇和短语.json")
sw5 = json_load("./stopwords/论文中常用的词汇和短语2.json")
sw6 = json_load("./stopwords/论文中常用的词汇和短语3.json")
sw7 = json_load("./stopwords/论文中常用的词汇和短语4.json")
sw8 = json_load("./stopwords/慎用.json")
sw9 = json_load("./stopwords/慎用2.json")

nlp = spacy.load("en_core_web_lg")
stopword = sw1 + sw2 + sw3 + sw4 + sw5 + sw6 + sw7 + sw8 + sw9
words_list = list()
for val in stopword:
    words = nlp(val)
    word_temp = list()
    for word in words:
        word_temp.append(word.lemma_.lower())
    if word_temp:  # 如果为空, 则放弃该关键词
        words_list.append("_".join(word_temp))
stopword = ["_".join(val.split(" ")) for val in words_list]
json_save(stopword, "../data/key_preprocess/paper_stopword.json")


