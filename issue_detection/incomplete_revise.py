import copy
import random

from prompt_temp_revise import *

random.seed(42)


def build_incomplete(datas, section_dict, field_dic):
    total_incomplete_datas = []

    def random_trim(s, num=0.3):
        length = len(s)
        remove_count = max(1, int(length * num))  # 至少去除1个字符
        remove_count = min(remove_count, length - 1)
        if random.choice([True, False]):
            return s[remove_count:]
        else:
            return s[:-remove_count]

    note_positives = []
    note_negatives = []
    for data in datas:
        raw_content = data["content"].replace("<sep1>", ":").replace("<sep2>", " ")
        sub_content_list = data["sub_contents"]
        if len(sub_content_list) == 1:
            sub_content = sub_content_list[0]["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
            # 这种情况语义都是完整的。这种情况预测的内容也可能有内容重复，但是与金标准相比，没有语义缺失，所以认为是完整的。
            temp = {"source": raw_content, "target": sub_content,
                    "sec_name": sub_content_list[0]["sub_meta"]["sec_name"],
                    "label": "完整",
                    "ctype": "Note-Content-Incomplete"}
            note_positives.append(temp)
        else:
            # 这种情况认为是完整的，因为raw_content是多个章节的杂糅所以相对金标注章节也是完整的。
            for sub_content in sub_content_list:
                if random.choice([True, False]):
                    target_content = sub_content["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
                    temp = {"source": raw_content, "target": target_content,
                            "sec_name": sub_content["sub_meta"]["sec_name"],
                            "label": "完整",
                            "ctype": "Note-Content-Incomplete"}
                    note_positives.append(temp)

        for sub_content in sub_content_list:
            fields = {key: value for key, value in sub_content["fields"].items() if value}

            if len(fields) > 4:
                target_str = ''.join(["{}".format(value) for key, value in fields.items() if key])
                # random_key = random.choice(list(fields.keys()))
                fields_1 = copy.deepcopy(fields)
                try:
                    sample_keys = random.sample(list(fields.keys()), k=random.randint(2, 6))
                    for random_key in sample_keys:
                        del fields_1[random_key]
                    if fields_1 and len(fields_1) > 2:
                        # 书暗处掉几个field
                        source_str = ''.join(["{}:{}".format(key, value) for key, value in fields_1.items() if key])
                        temp = {"source": target_str, "target": source_str,
                                "sec_name": sub_content["sub_meta"]["sec_name"],
                                "label": "不完整",
                                "ctype": "Note-Content-Incomplete"}
                        note_negatives.append(temp)
                except:
                    continue
        # 相同类型的章节，仅获取到其中一个，也算不完整

    print("note Content Incomplete samples:", len(note_positives), len(note_negatives))
    note_negatives = random.sample(note_negatives, min([1000, len(note_negatives)]))
    note_positives = random.sample(note_positives, min([1000, len(note_positives)]))  # 保持正负样本比例1:1

    totals = note_positives + note_negatives

    for data in totals:
        query = PROMPT_incomplate_Note.format(
            data["source"],
            data["target"],
            data["sec_name"],
            section_dict[data["sec_name"]]
        )
        answer = """```json[{{"语义是否完整": "{}"}}]```""".format(data["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": data["label"], "ctype": data["ctype"]}}
        total_incomplete_datas.append(temp)

    section_positives = []
    section_negatives = []

    # 字段内容内容缺失。
    total_sub_contents = []
    for data in datas:
        for sub_content in data["sub_contents"]:
            total_sub_contents.append(sub_content)
    for item in total_sub_contents:
        fields = item["fields"]
        fields = {k: v for k, v in fields.items() if v}

        for key, value in fields.items():
            # 缺少个别字符认为是完整的，缺少30%以上的字符认为是不完整的。
            source_str = item["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
            target_str = value
            temp = {"source": source_str, "target": target_str,
                    "sec_name": key,
                    "explain": field_dic[item["sub_meta"]["sec_name"]][key],
                    "label": "完整",
                    "ctype": "Section-Content-Incomplete"}
            section_positives.append(temp)

            # 缺少部分字符。随机丢到一两个单词。
            num = random.choice([0.2, 0.4, 0.6])
            target_str = random_trim(target_str, num)
            temp = {"source": source_str, "target": target_str,
                    "sec_name": key,
                    "explain": field_dic[item["sub_meta"]["sec_name"]][key],
                    "label": "不完整",
                    "ctype": "Section-Content-Incomplete"}
            section_negatives.append(temp)

    section_negatives = random.sample(section_negatives, 1000)
    section_positives = random.sample(section_positives, 1000)  # 保持正负样本比例1:1
    totals = section_positives + section_negatives
    for total in totals:
        query = PROMPT_incomplate_Section.format(total["source"],
                                                 total["sec_name"],
                                                 total["explain"],
                                                 total["target"])
        # answer = """```json[{{"语义是否完整": "{}", "情况说明": "[请补充该部分内容]"}}]```""".format(total["label"])
        answer = """```json[{{"语义是否完整": "{}"}}]```""".format(total["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": total["label"], "ctype": total["ctype"]}}
        total_incomplete_datas.append(temp)
    print("Section Content Incomplete samples:", len(section_positives), len(section_negatives))
    return total_incomplete_datas
