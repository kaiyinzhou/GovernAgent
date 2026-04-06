
import collections
import json

import random

random.seed(1124)


def read_datas(file):
    datas = json.load(open(file, 'r', encoding="utf-8"))
    return datas


def sample_data_for_check(datas):
    label_count_positive = collections.defaultdict(list)
    label_count_negative = collections.defaultdict(list)
    for item_datas in datas:
        for hos_name, datas in item_datas.items():
            for data in datas:
                data["human_label"] = data["errors"]
                error = data["errors"]
                query = data["queries"]
                for key, e in error.items():

                    if e == "杂糅":
                        label_count_positive["杂糅"].append({"query": query[key], "answer": "杂糅/不杂糅",
                                                             "predict": "杂糅"})
                    elif e == "不杂糅":
                        label_count_negative["不杂糅"].append({"query": query[key], "answer": "杂糅/不杂糅",
                                                             "predict": "不杂糅"})
                    elif e == "重复":
                        label_count_positive["重复"].append({"query": query[key], "answer": "重复/不重复",
                                                             "predict": "重复"})
                    elif e == "不重复":
                        label_count_negative["不重复"].append({"query": query[key], "answer": "重复/不重复",
                                                             "predict": "不重复"})
                    if e == "不正确":
                        label_count_positive["不正确"].append({"query": query[key], "answer": "正确/不正确",
                                                             "predict": "不正确"})
                    elif e == "正确":
                        label_count_negative["正确"].append({"query": query[key], "answer": "正确/不正确",
                                                             "predict": "正确"})
                    elif e == "不完整":
                        label_count_positive["不完整"].append({"query": query[key], "answer": "完整/不完整",
                                                             "predict": "不完整"})
                    elif e == "完整":
                        label_count_negative["完整"].append({"query": query[key], "answer": "完整/不完整",
                                                             "predict": "完整"})

    zarou_num = min([len(label_count_negative["不杂糅"]), len(label_count_positive["杂糅"]), 16])

    d1 = random.sample(label_count_negative["不杂糅"], k=zarou_num)
    d2 = random.sample(label_count_positive["杂糅"], k=zarou_num)
    print("杂糅：{}，不杂糅：{}".format(len(d2), len(d1)))

    chongfu_num = min([len(label_count_negative["不重复"]), len(label_count_positive["重复"]), 16])

    d3 = random.sample(label_count_positive["重复"], k=chongfu_num)
    d4 = random.sample(label_count_negative["不重复"], k=chongfu_num)
    print("重复：{}，不重复：{}".format(len(d3), len(d4)))

    zhengque_num = min([len(label_count_negative["正确"]), len(label_count_positive["不正确"]), 16])

    d5 = random.sample(label_count_positive["不正确"], k=zhengque_num)
    d6 = random.sample(label_count_negative["正确"], k=zhengque_num)
    print("正确：{}，不正确：{}".format(len(d5), len(d6)))

    wanzheng_num = min([len(label_count_negative["完整"]), len(label_count_positive["不完整"]), 16])
    d7 = random.sample(label_count_positive["不完整"], k=wanzheng_num)
    d8 = random.sample(label_count_negative["完整"], k=wanzheng_num)
    print("完整：{}，不完整：{}".format(len(d8), len(d7)))

    total_datas = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
    return total_datas


def write_json(datas, file):
    with open(file, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    for name in {"note", "section"}:
        total_datas = []
        if name == "note":
            files = [
                "./predicts/checker-note/note_errors_gold.json",
                "./predicts/checker-note/note_errors_predict.json",
            ]
        else:
            files = [
                "./predicts/checker-section/section_errors_gold.json",
                "./predicts/checker-section/section_errors_predict.json",
            ]
        for file in files:
            datas = read_datas(file)
            outs = sample_data_for_check([datas])
            total_datas.extend(outs)
        json.dump(total_datas, open(f"./for_label/{name}_error_datas.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=4)

#
