from function.func_ import *


# en_thesaurus.jsonl.txt处理
def synonym1():
    new_file = list()
    with open("./synonym/en_thesaurus.jsonl.txt", "r") as f:
        for line in f.readlines():
            line_ = json.loads(line.strip("\n"))
            new_file.append(line_)
    json_save(new_file, "./synonym/en_thesaurus.json")

    words = json_load("./synonym/en_thesaurus.json")
    synonyms = dict()
    for data_ in words:
        word = data_["word"]
        synonym = data_["synonyms"]
        synonyms.setdefault(word, list())
        synonyms[word].extend(synonym)
    json_save(synonyms, "./synonym/en_thesaurus_synonyms.json")


# th_en_US_new.js处理
def synonym2():
    data_ = json_load("./synonym/th_en_US_new.js")
    json_save(data_, "./synonym/th_en_US_new_synonyms.json")


# thesaurus-en.txt处理
def synonym3():
    new_file = dict()
    with open("./synonym/thesaurus-en.txt", "r") as f:
        for line in f.readlines():
            line_ = line.strip("\n")
            line_split = line_.split(",")
            if len(line_split) >= 1:
                new_file[line_split[0]] = line_split[1:]
    json_save(new_file, "./synonym/thesaurus-en_synonyms.json")


# 早期词表
def synonym4():
    new_file = dict()
    data = pd.read_csv("./synonym/早期词表.csv", header=None)
    data_index = data.index
    num = 0
    for ind in data_index:
        main_word = data.loc[ind, 0]
        syno1 = data.loc[ind, 1]
        syno2 = data.loc[ind, 2]
        new_file.setdefault(main_word, list())
        if not pd.isna(syno1):
            new_file[main_word].extend(syno1.strip('"').split(", "))
        if not pd.isna(syno2):
            new_file[main_word].extend(syno2.strip('"').split(", "))
    json_save(new_file, "./synonym/早期词表_synonyms.json")


# 整合
en_thesaurus = json_load("./synonym/en_thesaurus_synonyms.json")
th_en_US_new = json_load("./synonym/th_en_US_new_synonyms.json")
thesaurus_en = json_load("./synonym/thesaurus-en_synonyms.json")
early_word = json_load("./synonym/早期词表_synonyms.json")

synonym_all = dict()


def dict_concat(dict_all: dict, dict_: dict):
    for key_, value_ in dict_.items():
        dict_all.setdefault(key_, list())
        dict_all[key_].extend(value_)
        dict_all[key_] = list(set(dict_all[key_]))


dict_concat(synonym_all, en_thesaurus)
dict_concat(synonym_all, th_en_US_new)
dict_concat(synonym_all, thesaurus_en)
dict_concat(synonym_all, early_word)

json_save(synonym_all, "../data/key_preprocess/synonym_word.json")



