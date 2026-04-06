# -*- coding:utf-8 -*-

import random
import json
from prompt_temp_revise import *

random.seed(42)


def build_conflation(datas, section_str, field_str):
    total_conflation_datas = []
    note_positives = []
    note_negatives = []
    for data in datas:
        raw_content = data["content"].replace("<sep1>", ": ").replace("<sep2>", " ")
        # 首先将原始就杂糅的拿出来作正负样本。
        sub_section_dicts = data["sub_contents"]
        help_labels = [item["sub_meta"]["sec_name"] for item in sub_section_dicts]

        if len(sub_section_dicts) >= 2:
            temp = {"source": raw_content,
                    "label": "杂糅",
                    "source_type": "type1",
                    "helps": help_labels, "ctype": "Note-conflation"}
            note_positives.append(temp)
        else:
            print(data)
            print(raw_content)
            temp = {"source": raw_content, "label": "不杂糅",
                    "source_type": "type2",
                    "helps": help_labels, "ctype": "Note-conflation"}
            note_negatives.append(temp)

    # 从datas中随机采样两到三个样本，然后构建正样本。
    total_sub_contents = []
    for data in datas:
        for sub_content in data["sub_contents"]:
            total_sub_contents.append(sub_content)
    # 多个不同的记录混合在一起，属于杂糅
    for _ in range(200):
        sampled_contents = random.sample(total_sub_contents, k=random.randint(2, 4))
        # 构建正样本
        if len(sampled_contents) >= 2:
            content_list = [sample["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ") for sample in
                            sampled_contents]

            content_str = "".join(content_list)
            helps = [sample["sub_meta"]["sec_name"] for sample in sampled_contents]
            positive_sample = {"source": content_str, "label": "杂糅",
                               "source_type": "type4",
                               "helps": helps, "ctype": "Note-conflation"}
            note_positives.append(positive_sample)

    # note_negatives = random.sample(note_negatives, int(len(note_positives) * 2))  # 保持正负样本比例1:1
    print("Note conflation samples:", len(note_positives), len(note_negatives))
    totals = note_positives + note_negatives

    for item in totals:
        query = PROMPT_Conflation_Note.format(item["source"], section_str)
        # answer = """```json[{{"是否杂糅": "{}", "情况说明": "[请补充该部分内容]"}}]```""".format(item["label"])
        answer = """```json[{{"是否杂糅": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "helps": item["helps"], "ctype": item["ctype"]}}
        total_conflation_datas.append(temp)
    # 构建section层面的杂糅数据。
    # 直接拼接构建
    section_positives = []
    section_negatives = []
    for data in datas:
        for sub_content in data["sub_contents"]:
            sub_content_str = sub_content["sub_content"].replace("<sep1>", ": ").replace("<sep2>", " ")
            note_name = sub_content["sub_meta"]["sec_name"]
            fields = sub_content["fields"]
            fields = {k: v for k, v in fields.items() if v}  # 过滤掉空字段，并替换掉分隔符
            # 直接拼接，任意其中的一个field都不杂糅。
            for key, field in fields.items():
                temp_str = "{}".format(field)
                negative_sample = {
                    "note_name": note_name,
                    "note_content": sub_content_str,
                    "sec_name": key,
                    "sec_content": temp_str,
                    "label": "不杂糅",
                    "source_type": "type3",
                    "gold_field": fields,
                    "helps": [key],
                    "ctype": "section-conflation"}

                section_negatives.append(negative_sample)
            # 选择任意2-3个拼接认为是杂糅
            for key, field in fields.items():
                if len(fields.keys()) <= 2:
                    conf_keys = fields.keys()
                    conf_fields = [fields[k] for k in conf_keys]
                else:
                    conf_keys = random.sample([k for k in fields.keys() if k != key], k=min([2, 3]))
                    conf_fields = [fields[k] for k in conf_keys]
                temp_str = "{}".format(field) + "".join(["{}".format(k) for k in conf_fields if k])
                positive_sample = {
                    "note_name": note_name,
                    "note_content": sub_content_str,
                    "sec_name": key,
                    "sec_content": temp_str,
                    "label": "杂糅",
                    "source_type": "type3",
                    "gold_field": fields,
                    "ctype": "section-conflation"}
                section_positives.append(positive_sample)

    section_negatives = random.sample(section_negatives, min([1000, len(section_negatives)]))
    section_positives = random.sample(section_positives, min([1000, len(section_positives)]))
    totals = section_positives + section_negatives

    for item in totals:
        query = PROMPT_Conflation_Section.format(
            item["note_name"],
            item["note_content"],
            item["sec_name"],
            item["sec_content"],
            json.dumps(field_str[item["note_name"]], ensure_ascii=False)
        )
        # answer = """```json[{{"是否杂糅": "{}", "情况说明": "[请补充该部分内容]"}}]```""".format(item["label"])
        answer = """```json[{{"是否杂糅": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "gold_field": item["gold_field"], "ctype": item["ctype"]}}
        total_conflation_datas.append(temp)
    print("Note conflation samples:", len(section_positives), len(section_negatives))
    return total_conflation_datas
