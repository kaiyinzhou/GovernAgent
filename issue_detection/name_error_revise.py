# -*- coding:utf-8 -*-

import json
import random

from prompt_temp_revise import *

random.seed(42)


def build_name_error(datas, section_str, field_str):
    # 构建名称错误数据。
    # 章节名称错误，章节杂糅时候认为名称错误
    total_name_errors = []
    note_positives = []
    note_negatives = []
    for data in datas:
        raw_content = data["content"].replace("<sep1>", ":").replace("<sep2>", " ")
        sec_name = data["meta"]["sec_name"]
        sub_content_list = data["sub_contents"]
        if len(sub_content_list) == 1:
            sub_content = sub_content_list[0]["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
            sub_sec_name = sub_content_list[0]["sub_meta"]["sec_name"]
            if sub_sec_name == sec_name:
                temp = {"source": sub_content, "ask_name": sub_sec_name, "label": "正确",
                        "helps": [sub_sec_name], "ctype": "Note-Name-error"}
                note_positives.append(temp)
                ask_name = random.sample(section_str.keys(), k=1)[0]
                if ask_name != sub_sec_name:
                    temp = {"source": sub_content, "ask_name": ask_name, "label": "不正确",
                            "helps": [sub_sec_name], "ctype": "Note-Name-error"}
                    note_negatives.append(temp)
            else:
                temp = {"source": raw_content, "ask_name": sub_sec_name, "label": "不正确",
                        "helps": [sub_sec_name], "ctype": "Note-Name-error"}
                note_negatives.append(temp)
        else:
            for sub_content_dict in sub_content_list:
                sub_content = sub_content_dict["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
                sub_sec_name = sub_content_dict["sub_meta"]["sec_name"]
                temp = {"source": sub_content, "ask_name": sub_sec_name, "label": "正确",
                        "helps": [sub_sec_name], "ctype": "Note-Name-error"}
                note_positives.append(temp)
                ask_name = random.sample(section_str.keys(), k=1)[0]
                if ask_name != sub_sec_name:
                    temp = {"source": sub_content, "ask_name": ask_name, "label": "不正确",
                            "helps": [sub_sec_name], "ctype": "Note-Name-error"}
                    note_negatives.append(temp)
        total_sub_contents = []
        for data in datas:
            for sub_content in data["sub_contents"]:
                total_sub_contents.append(sub_content)
        for _ in range(5):
            sampled_contents = random.sample(total_sub_contents, k=random.randint(2, 3))
            content_list = [sample["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
                            for sample in sampled_contents]
            content_str = "".join(content_list)
            helps = [sample["sub_meta"]["sec_name"] for sample in sampled_contents]
            ask_name = random.sample(helps, k=1)[0]
            if len(set(helps)) == 1:
                positive_sample = {"source": content_str, "ask_name": ask_name, "label": "正确", "helps": helps,
                                   "ctype": "Note-Name-error"}
                note_positives.append(positive_sample)
            else:
                negative_sample = {"source": content_str, "ask_name": ask_name, "label": "不正确", "helps": helps,
                                   "ctype": "Note-Name-error"}
                note_negatives.append(negative_sample)
    note_positives = random.sample(note_positives, min([1000, len(note_positives)]))
    note_negatives = random.sample(note_negatives, min([1000, len(note_negatives)]))
    totals = note_positives + note_negatives
    for item in totals:
        query = PROMPT_Name_Error_Note.format(item["source"],
                                              item["ask_name"],
                                              json.dumps(section_str, ensure_ascii=False))
        answer = """```json[{{"是否正确": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "helps": item["helps"], "ctype": item["ctype"]}}
        total_name_errors.append(temp)

    print("Name Error samples:", len(note_positives), len(note_negatives))
    section_positives = []
    section_negatives = []
    total_sub_contents = []
    for data in datas:
        for sub_content in data["sub_contents"]:
            total_sub_contents.append(sub_content)
    # 随机采样一个作为正负样本。
    for item in total_sub_contents:
        note_name = item["sub_meta"]["sec_name"]
        fields = item["fields"]
        fields = {k: v for k, v in fields.items() if v}

        for key, value in fields.items():
            temp = {
                "sec_content": value,
                "sec_name": key,
                "sec_explain": field_str[note_name][key],
                "total_labels": field_str,
                "label": "正确",
                "helps": [key],
                "field_str": field_str,
                "ctype": "Section-Name-error"}
            section_positives.append(temp)

            ask_name = random.sample(fields.keys(), k=1)[0]
            if ask_name != key and value != fields[ask_name]:  # 如果采样的名称和当前字段名称不一致
                temp = {
                    "sec_content": value,
                    "sec_name": ask_name,
                    "sec_explain": field_str[note_name][ask_name],
                    "total_labels": field_str,
                    "label": "不正确",
                    "helps": [key],
                    "ctype": "Section-Name-error"
                }
                section_negatives.append(temp)
        # 构建混合字段
        for _ in range(10):
            sampled_key = random.sample(fields.keys(), k=random.randint(min([2, len(fields.keys())]),
                                                                        min([4, len(fields.keys())])))
            if len(set(sampled_key)) == 1:
                continue

            sampled_fields = [fields[key] for key in sampled_key]
            try:
                temp = {
                    "sec_content": "".join(sampled_fields),
                    "sec_name": sampled_key[0],
                    "sec_explain": field_str[note_name][sampled_key[0]],
                    "total_labels": field_str[note_name],
                    "label": "不正确",
                    "helps": sampled_key,
                    "ctype": "Section-Name-error"
                }
                if "非结构化大文本" in temp["sec_explain"]:
                    temp["label"] = "正确"
                section_negatives.append(temp)
            except:
                continue

        # 重复字段
        if fields:
            sampled_key = random.sample(fields.keys(), k=1)[0]
            temp = {"sec_content": fields[sampled_key] + fields[sampled_key],
                    "sec_name": sampled_key,
                    "sec_explain": field_str[note_name][sampled_key],
                    "total_labels": field_str[note_name],
                    "label": "正确",
                    "helps": [sampled_key],
                    "ctype": "Section-Name-error"}

            section_positives.append(temp)

    section_positives = random.sample(section_positives, min([1000, len(section_positives)]))
    section_negatives = random.sample(section_negatives, min([1000, len(section_negatives)]))
    totals = section_negatives + section_positives
    for item in totals:
        query = PROMPT_Name_Error_Section.format(item["sec_content"],
                                                 item["sec_name"],
                                                 item["sec_explain"],
                                                 json.dumps(item["total_labels"], ensure_ascii=False))
        # answer = """```json[{{"是否正确": "{}", "情况说明": "[请补充该部分内容]"}}]```""".format(item["label"])
        answer = """```json[{{"是否正确": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "helps": item["helps"], "ctype": item["ctype"]}}
        total_name_errors.append(temp)

    print("Section Name Error samples:", len(section_positives), len(section_negatives))
    return total_name_errors
