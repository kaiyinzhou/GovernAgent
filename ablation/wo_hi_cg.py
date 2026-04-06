# -*- coding:utf-8 -*-
"""
串联推理，先推理完成章节，再推理字段
"""
import copy
import random

import tqdm

from utils import *

section_explain = json.load(open("../configs/label_explain.json", encoding='utf-8'))["Section"]
field_explain = json.load(open("../configs/label_explain.json", encoding='utf-8'))["Field"]


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
    random.shuffle(datas)
    return datas[:5000]


class PiplineInference:
    def __init__(self, model_name, url, ctype="extract"):
        self.model_name = model_name
        self.url = url
        self.ctype = ctype

    # async def tool(self, querys):
    #     batch_results = await async_process_queries(querys, self.url, self.model_name)
    #     return batch_results

    async def batch_inference(self, datas):
        """
        批量推理
        :param datas: 输入数据列表
        :return: 推理结果列表
        """
        total_outs = []
        batch_size = 16
        total_batch_datas = [datas[index:index + batch_size] for index in range(0, len(datas), batch_size)]
        for batch_datas in tqdm.tqdm(total_batch_datas, desc="Batch Inference"):
            batch_datas = await self.section_inference(batch_datas)
            batch_datas = await self.field_inference(batch_datas)
            total_outs.extend(batch_datas)
        return total_outs

    async def section_inference(self, datas):
        """
        对章节进行推理
        :param datas: 输入数据列表
        :return: 推理结果列表
        """
        for data in datas:
            meta = data["meta"]
            sub_content = data["content"].replace("<sep1>", ":").replace("<sep2>", " ")
            data["sub_contents"] = [{"sub_content": sub_content, "sub_meta": meta}]
        return datas

    def _function_field_extract(self, answers):
        answers = json.loads(answers)["choices"][0]["message"]["content"]
        if answers.startswith("["):
            answers = "```json" + answers
        if answers.endswith("]"):
            answers += "```"
        try:
            answers = re.findall(r"```json([\s\S]*)```", answers.split("</think>")[-1], re.S)[0]
            answers = eval(answers.strip())
            # answers = eval(answers)
            if isinstance(answers, dict):
                answers = [answers]
            contents = []
            for answer in answers:
                if "章节内容" in answer:
                    sub_content = answer["章节内容"]
                else:
                    sub_content = answer["字段内容"]
                contents.append(sub_content)
            return contents
        except:
            try:
                answers = re.findall(r"\"章节内容\"\: \"(.*?)$", answers, re.S)
                return answers
            except:
                answers = []
                return answers

    def _function_agent_field(self, raw_content, answers):
        answers = json.loads(answers)["choices"][0]["message"]["content"]
        if answers.startswith("["):
            answers = "```json" + answers
        if answers.endswith("]"):
            answers += "```"
        if answers.startswith("```["):
            answers = answers.replace("```[", "```json[")
        answers = re.findall(r"```json([\s\S]*)```", answers.split("</think>")[-1], re.S)[0]

        answers = eval(answers)
        if isinstance(answers, dict):
            answers = [answers]
        contents = []
        for answer in answers:
            action_content = ""
            for key, value in answer.items():
                if "动作" in key and "生成" in value:
                    action_content = answer["生成内容"]
                    # TODO 生成动作生成的内容不计算幻觉率
                if "动作" in key and "拷贝" in value:
                    start = answer["拷贝起始字符"].strip()
                    end = answer["拷贝终止字符"].strip()
                    length = answer["拷贝长度"]
                    action_content = find_closest_substring(raw_content, start, end, int(length))
            if not action_content:
                return []
            # if action_content:
            # if action_content not in {"未抽取到相关内容"}:
            if action_content:
                if isinstance(action_content, list):
                    action_content = "##".join(action_content)
                contents.append(action_content)
        return ['##'.join(contents)]

    async def field_inference(self, datas):
        """
        对字段进行推理
        :param datas: 输入数据列表
        :return: 推理结果列表

        """
        PROMPT_extract = """你是一个病历字段抽取智能体，请完成下面的任务。
请从下面的“{}”病历中抽取“{}”字段的内容，“{}”是“{}”。
抽取时请注意以下事项：
（1）当字段内容重复出现时，只需抽取一次。
（2）对于病历中的乱码部分，不需要抽取。
（3）当需要抽取的内容并非连续出现时，请确保抽取内容完整，不连续的内容用“##”号连接。
（4）按照下面的格式进行抽取：```json[{{"字段名称": "_", "字段内容": "_"}}]```。
（5）当需要抽取的字段在病历文本中不存在时，“字段内容”部分填写“未抽取到相关内容”。
病历文本：
{}"""
        PROMPT_agent = """你是一个病历字段抽取智能体，请完成下面的任务。
请从下面的“{}”病历中抽取“{}”字段的内容， “{}”是“{}”。

抽取时请注意以下事项：
（1）通过使用“拷贝”和“生成”动作完成字段内容抽取。
其中，“拷贝”动作指的是你需要发出“拷贝命令”，并同时给出“拷贝起始字符”、“拷贝终止字符”、“拷贝长度”参数，从而实现部分字段内容的抽取。
其中拷贝起始字符是章节的开始标志，拷贝终止字符是章节的结束标志，拷贝起始位置和终止位置通常包含5-8个字符。拷贝长度是从拷贝起始位置到终止位置的字符长度。
当遇到乱码、重复、杂糅需要跳过某些内容时需要在跳过位置停止拷贝，然后重新寻找拷贝起始位置，并发出“拷贝命令”。
当需要处理乱码或待提取的字段内容较短时（小于20 tokens），需要使用“生成”动作，即你需要发出“生成命令”，并直接生成目标内容，从而实现字段内容的抽取。
不允许重复拷贝或生成相同内容。
（2）按照下面的格式依次输出拷贝和生成过程：
```json[{{"动作1": "拷贝", "拷贝起始字符": "_", "拷贝终止字符": "_", "拷贝长度": "_"}},
{{"动作2": "生成", "生成内容": "_"}},
{{"动作3": "拷贝", "拷贝起始字符": "_", "拷贝终止字符": "_", "拷贝长度": "_"}}, ...]```
（3）当需要抽取的字段在病历文本中不存在时，返回```json[{{"动作1": "生成", "生成内容": "未抽取到相关内容"}}]```。

病历文本：
{}"""
        PROMPT = PROMPT_extract if "extract" in self.ctype else PROMPT_agent
        batch_query = []
        batch_fields_names = []
        batch_datas = []
        batch_index = []
        batch_raw_content = []
        for data in datas:
            sub_contents = data["sub_contents"]
            for index, sub_content_dic in enumerate(sub_contents):
                data["sub_contents"][index]["fields"] = {}
                sub_content = sub_content_dic["sub_content"]
                sec_name = sub_content_dic["sub_meta"]["sec_name"]
                for field_name in field_explain[sec_name]:
                    query = PROMPT.format(sec_name, field_name, field_name,
                                          field_explain[sec_name][field_name],
                                          sub_content.replace("<sep1>", ":").replace("<sep2>", " "))
                    batch_query.append(query)
                    batch_fields_names.append(field_name)
                    batch_datas.append(data)
                    batch_index.append(index)
                    batch_raw_content.append(sub_content.replace("<sep1>", ":").replace("<sep2>", " "))
        batch_predicts = await async_process_queries(batch_query, self.url, self.model_name)

        for data, predict, field_name, index, raw_content in zip(batch_datas, batch_predicts,
                                                                 batch_fields_names, batch_index, batch_raw_content):
                if predict:

                    if self.ctype == "extract":
                        field_content = self._function_field_extract(predict)
                    elif self.ctype == "agent":
                        field_content = self._function_agent_field(raw_content, predict)
                    else:
                        print("ERROR")
                        return
                    field_content = "##".join(field_content).replace("未抽取到相关内容", "")
                    data["sub_contents"][index]["fields"][field_name] = field_content
        return datas


def write_results_to_file(results, output_file):
    """
    将推理结果写入文件
    :param results: 推理结果列表
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 示例用法,这个推理脚本是不做HI结构推理的。直接做拷贝生成或直接生成
    import asyncio

    # model_name = "qwen-extract-14b"
    # data_ctype = "test"

    # model_name = "extract-qwen-sft-7b"
    # # model_name = "extract-llama-sft-8b"
    # data_ctype = "jiwai"
    model_name = "extract-7b-wo-hi"
    data_ctype = "test"

    url = "http://10.0.0.230:8833/v1"

    if data_ctype == "test":
        datas = read_datas("./datas/total_datas_10000.jsonl")
    else:
        datas = read_datas("./datas/total_datas_10000.jsonl")

    agent = PiplineInference(model_name, url, "agent")
    extract = PiplineInference(model_name, url, "extract")
    loop = asyncio.get_event_loop()
    os.makedirs("./predicts/{}/".format(model_name), exist_ok=True)
    batch_datas = [datas[index:index + 128] for index in range(0, len(datas), 128)]
    with open("./predicts/{}/{}_results.json".format(model_name, data_ctype), 'w', encoding='utf-8') as f:
        for batch_data in batch_datas:
            extract_results = loop.run_until_complete(extract.batch_inference(batch_data))
            print(extract_results[-1])
            for agent_result in extract_results:
                f.write(json.dumps(agent_result, ensure_ascii=False) + '\n')

