import copy
import random

from prompt_temp_revise import *

random.seed(42)
import json

def build_duplication(datas):
    def random_trim(s, num=0.3):
        length = len(s)
        remove_count = max(1, int(length * num))  # 至少去除1个字符
        remove_count = min(remove_count, length - 1)
        if random.choice([True, False]):
            return s[remove_count:], s[:-remove_count]  # 返回去除部分和保留部分
        else:
            return s[:-remove_count], s[remove_count:]  # 返回保留部分和去除部分

    note_positives = []
    note_negatives = []
    total_duplication_datas = []
    for data in datas:
        raw_content = data["content"].replace("<sep1>", ":").replace("<sep2>", " ")

        sub_content_list = data["sub_contents"]
        for section in sub_content_list:
            sub_content = section["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
            # 该字段直接使用，作为不重复样本
            temp = {"source": sub_content, "label": "不重复", "ctype": "Note-Duplication"}
            note_positives.append(temp)
            # 将该字段重复一次，作为重复样本
            temp = {"source": sub_content + sub_content, "label": "重复", "ctype": "Note-Duplication"}
            note_negatives.append(temp)
            # 将该字段与field的拼接内容重复一次，作为重复样本
            fields_str = ''.join(["{}:{}".format(key, value) for key, value in section["fields"].items()])
            temp = {"source": sub_content + fields_str, "label": "重复", "ctype": "Note-Duplication"}
            note_negatives.append(temp)

            # 将原来的raw_content与sub_content拼接作为重复样本
            temp = {"source": raw_content + sub_content, "label": "重复", "ctype": "Note-Duplication"}
            note_negatives.append(temp)

            # 将该字段与field的拼接起来不算重复，重复其中个别的key也不算重复
            fields_str = ''.join(["{}:{}".format(key, value) for key, value in section["fields"].items()])
            temp = {"source": fields_str, "label": "不重复", "ctype": "Note-Duplication"}
            note_positives.append(temp)

            # 其中1-3个field重复，不能算作section重复
            for _ in range(30):
                try:
                    sampled_keys = random.sample(list(section["fields"].keys()), k=random.randint(1, 3))
                    if len(sampled_keys) < len(list(section["fields"].keys())):
                        fields_sampled_list = ["{}:{}".format(key, section["fields"][key]) for key in sampled_keys]
                        fields_raw_list = ["{}:{}".format(key, value) for key, value in section["fields"].items()]
                        field_list = fields_raw_list + fields_sampled_list
                        # random.shuffle(field_list)
                        fields_str = ''.join(field_list)
                        temp = {"source": fields_str, "label": "不重复", "ctype": "Note-Duplication"}
                        note_positives.append(temp)
                except:
                    continue

    note_negatives = random.sample(note_negatives, min([1000, len(note_negatives)]))
    note_positives = random.sample(note_positives, min([1200, len(note_positives)]))

    print("Note Duplication samples:", len(note_positives), len(note_negatives))
    totals = note_positives + note_negatives
    for item in totals:
        query = PROMPT_Duplication_Note.format(item["source"])
        # answer = """```json[{{"是否重复": "{}", "情况说明": "[请补充该部分内容]"}}]```""".format(item["label"])
        answer = """```json[{{"是否重复": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "ctype": item["ctype"]}}
        total_duplication_datas.append(temp)

    # 字段重复情况
    section_positives = []
    section_negatives = []
    for data in datas:
        # 将data["content"]转换为字典
        content_list = {}
        for item in data["content"].split("<sep2>"):
            key, value = item.split("<sep1>")
            content_list[key] = value

        sub_content_list = data["sub_contents"]
        for sub_content in sub_content_list:
            fields = sub_content["fields"]
            fields = {k: v for k, v in fields.items() if v}  # 过滤掉空字段
            for key, value in fields.items():
                # 该字段直接使用，作为不重复样本
                new_fields = copy.deepcopy(fields)
                new_fields = {k: v for k, v in new_fields.items()}  # 过滤掉空字段
                temp = {"sec_name": key,
                        "sec_content": value,
                        "label": "不重复",
                        "fields": new_fields,
                        "source_ctype": "type1",
                        "ctype": "Section-Duplication"}
                section_positives.append(temp)
                # 选1个key,重复2-3次

                fields_temp = copy.deepcopy(fields)
                for _ in range(random.randint(2, 3)):
                    new_key = random.choice(list(fields_temp.keys()))
                    if random.choice([True, False]):
                        fields_temp[new_key] = value + fields_temp[new_key]
                    else:
                        fields_temp[new_key] = fields_temp[new_key] + value

                temp = {"sec_name": key,
                        "sec_content": value,
                        "label": "重复",
                        "fields": fields_temp,
                        "source_ctype": "type2",
                        "ctype": "Section-Duplication"}

                section_negatives.append(temp)

                # 截取value_str中的一部分内容，随机放置到其他字段上。
                fields_temp = copy.deepcopy(fields)
                test_key = random.choice(list(fields_temp.keys()))
                value_str = fields_temp[test_key]
                sub_value_str1, sub_value_str2 = random_trim(value_str, num=random.choice([0.5, 0.6]))
                fields_temp[random.choice(list(fields_temp.keys()))] += sub_value_str1
                temp = {
                        "sec_name": test_key,
                        "sec_content": sub_value_str1,
                        "label": "重复",
                        "fields": fields_temp,
                        "source_ctype": "type3",
                        "ctype": "Section-Duplication"}
                section_negatives.append(temp)

    section_positives = random.sample(section_positives, 1000)
    section_negatives = random.sample(section_negatives, 1000)
    print("Section Duplication samples:", len(section_positives), len(section_negatives))
    totals = section_positives + section_negatives
    for item in totals:
        query = PROMPT_Duplication_Section.format(
            item["sec_name"],
            item["sec_content"],
            json.dumps(item["fields"], ensure_ascii=False)
        )
        answer = """```json[{{"是否重复": "{}"}}]```""".format(item["label"])
        temp = {"lg": "ch", "conversations": [{"content": query, "role": "user"},
                                              {"content": answer, "role": "assistant"}],
                "meta": {"label": item["label"], "fields": item["fields"], "ctype": item["ctype"],
                         "source_ctype": item.get("source_ctype", "")}}
        total_duplication_datas.append(temp)
    return total_duplication_datas
