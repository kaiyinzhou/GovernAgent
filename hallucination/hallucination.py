# -*- coding:utf-8 -*-

import collections
import json
import os
import re
import asyncio

from utils import async_process_queries


def read_datas(file):
    """
    从指定文件中读取数据，返回一个包含所有数据的列表
    :param file: 文件路径
    :return: 数据列表
    """
    with open(file, 'r', encoding='utf-8') as rf:
        datas = []
        for line in rf:
            data = json.loads(line)
            datas.append(data)
    return datas


def punction_split(text):
    """
    使用标点符号将文本切分成短片段
    :param text: 输入文本
    :return: 切分后的短片段列表
    """
    import re
    # 使用正则表达式匹配中文标点符号和英文标点符号进行切分
    segments = re.split(r'[，。！？；：,!?;:\n \"\\#]', text)
    return [seg.strip() for seg in segments if seg.strip()]


async def tool(querys, url, model_name):
    return await async_process_queries(querys, url, model_name)


def parser_predict(predict, target_key):
    predict = json.loads(predict)["choices"][0]["message"]["content"]
    # 定位 markdown代码块中的json内容
    try:

        json_content = re.findall("\{[\s\S]*\}", predict, re.S)

        # 尝试解析json数据
        data = json.loads(json_content[0])
        result = data[target_key]
        if result in ("严重幻觉", "普通幻觉"):
            return result
        else:
            # 格式不正确或键不存在，默认返回"否"
            return "普通幻觉"
    except:
        # 出现解析错误，同样稳健返回默认“否”
        return "普通幻觉"


async def extract_hallucination(datas):
    """
    提取幻觉片段
    :param datas: 数据列表
    :return: 幻觉片段列表
    """
    hallucination_segments = []
    total_segments = 0
    for i, data in enumerate(datas):
        sub_contents = data["sub_contents"]
        for sub_content in sub_contents:
            sub_content_str = sub_content["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
            predict_fields = sub_content["fields"]
            for field_name, value in predict_fields.items():
                if value:
                    segments = punction_split(value)
                    total_segments+=len(segments)
                    for segment in segments:
                        # 检查该片段是否在原始文本中
                        if segment not in sub_content_str and segment not in {"包括姓名、性别、年龄等", "患者基本信息"}:
                            # 如果不在原始文本中，则认为是幻觉片段
                            if field_name not in {"患者基本信息"}:

                                hallucination_segments.append({
                                    "index": i,
                                    "section": sub_content["sub_meta"]["sec_name"],
                                    "field": field_name,
                                    "hallucination_segment": segment,
                                    "original_text": sub_content_str
                                })
    print(f"Total hallucination segments found: {len(hallucination_segments)}")
    # 调用大语言模型分析幻觉
    PROMPT = """在大语言模型应用中，有一类幻觉是指模型生成的内容与输入内容不相符。
现在有下面一段幻觉片段，请基于原始文本内容，判断这段幻觉片段属于的幻觉类型，有两种可以选择的类型：
（1）严重幻觉：对诊疗内容的虚构或捏造，可能会导致潜在的安全风险和误导性信息传播。
例如：篡改检查检验结果或单位，虚构患者症状表现或诊断结果，捏造治疗方案或用药信息，以及虚构患者基本信息等。
（2）普通幻觉：异常添加的内容，可能会导致信息的冗余或不必要的复杂性，但不会直接影响诊疗安全。例如：添加无医疗意义的单词或短语。
按照下面的格式输出结果：
```json[
{{"幻觉类型": "严重幻觉/普通幻觉", "原因解释": "XXXX"}}
]```

## 幻觉片段
{}

## 参考文本
{}"""
    total_query = []
    for segment in hallucination_segments:
        hallucination_segment = segment["hallucination_segment"]
        original_text = segment["original_text"]
        query = PROMPT.format(hallucination_segment, original_text)
        total_query.append(query)

    total_batch_predicts = await tool(total_query, url, model_name)
    error_type = collections.defaultdict(int)
    for segment, predict in zip(hallucination_segments, total_batch_predicts):
        predict = parser_predict(predict, "幻觉类型")
        if predict == "严重幻觉":
            error_type["严重幻觉"] += 1
            print(f"严重幻觉: {segment['hallucination_segment']}")
            print(f"原始文本: {segment['original_text']}")
            segment["hallucination_type"] = "严重幻觉"
        elif predict == "普通幻觉":
            error_type["普通幻觉"] += 1
            print(f"普通幻觉: {segment['hallucination_segment']}")
            print(f"原始文本: {segment['original_text']}")
            segment["hallucination_type"] = "普通幻觉"
        else:
            error_type["未知"] += 1
            segment["hallucination_type"] = "未知"
    print(f"Hallucination types found: {error_type}")
    for ctype, num, in error_type.items():
        print(f"幻觉类型: {ctype}, 数量: {num}")
        print(f"{ctype}: {num}/{total_segments}")


    return hallucination_segments


if __name__ == "__main__":
    url = "http://10.0.0.231:8811/v1"
    model_name = "qwen2.5-70b"

    # base_model = "goveragent-7b"
    # base_model = "goveragent-14b-wo-hi"
    base_model = "extract-7b-wo-hi"
    files = [
        "./predicts/{}/test_results.json".format(base_model),
    ]

    loop = asyncio.get_event_loop()
    for file in files:
        datas = read_datas(file)
        hallucination_segments = loop.run_until_complete(extract_hallucination(datas))
        os.makedirs(file.replace("./predicts", "./ha_out").replace("test_results.json", ""), exist_ok=True)

        json.dump(hallucination_segments, open(file.replace("./predicts", "./ha_out").
                                               replace("labeled", "predicts"), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
