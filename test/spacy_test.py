import spacy
import re
# 激活GPU
spacy.prefer_gpu()
# 加载英文数据处理
nlp = spacy.load("en_core_web_lg")
doc = nlp("This is a sentence._ I am god!")
for sent in doc.sents:
    for word in sent:
        if "".join(set(re.sub(re.compile(r'x+', re.I), "", word.shape_))) in ["", "_", "."]:
            if not word.is_stop and word != "." and word != "_":
                print(word.pos_)
                word = word.lemma_.lower()
                print(word)
# sent_ = [sent for sent in doc.sents]
# print(sent_)
# print(doc.noun_chunks)
# for val in doc.sentiment:
#     print(val)



import spacy
# from spacy import displacy
#
# nlp = spacy.load("en_core_web_lg")
# doc = nlp("Apple is looking at buying uk startup for $1 billion.")
# for token in doc:
#     # text: 原本文本
#     # lemma_: 词原型
#     # pos_: 简单的词性标签
#     # tag_: 详细的词性标签
#     # dep_: 句法依赖
#     # shape_: 词形状--大写 | 标点符号 | 数字
#     # is_alpha: 词是否为alpha字符
#     # is_stop: 词是否为停用词|常用词
#     print([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#           token.shape_, token.is_alpha, token.is_stop])
#     # test: 原本文本
#     # has_vector: 词汇是否有词向量
#     # vector_norm: 词汇词向量的L2范数
#     # is_oov: 词汇是否为词表之外
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)
#
# for ent in doc.ents:
    # text: 原文本
    # start_char: 词开始位置
    # end_char: 词结束位置
    # label_: 词的命名实体, 如公司|国家|货币等
    # print(ent.text, ent.start_char, ent.end_char, ent.label_)



import spacy
from spacy.matcher import Matcher

# nlp = spacy.load("en_core_web_lg")
# matcher = Matcher(nlp.vocab)
# # Add match ID "HelloWorld" with no callback and one pattern
# pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
# matcher.add("HelloWorld", [pattern])
#
# doc = nlp("Hello, world! Hello world!")
# matches = matcher(doc)
# for match_id, start, end in matches:
#     string_id = nlp.vocab.strings[match_id]  # Get string representation
#     span = doc[start:end]  # The matched span
#     print(match_id, string_id, start, end, span.text)
