# -*- coding:utf-8 -*-

import collections
import re

import tqdm

from prompt_temp import *
from utils import *

section_explain = json.load(open("./configs/label_explain_en.json", encoding='utf-8'))["Section"]
field_explain = json.load(open("./configs/label_explain_en.json", encoding='utf-8'))["Field"]
experience_add = json.load(open("./configs/experience.json", encoding='utf-8'))
# experience_add = json.load(open("./configs/new_experiences_en.json", encoding='utf-8'))


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


class PiplineInference:
    def __init__(self, model_name, url,
                 model_name1, url1,
                 ctype="extract"):
        self.model_name = model_name
        self.url = url
        self.model_name1 = model_name1
        self.url1 = url1
        self.ctype = ctype

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

    def _function_agent_field(self, raw_content, answers, mode):
        answers = json.loads(answers)["choices"][0]["message"]["content"]
        if answers.startswith("["):
            answers = "```json\n" + answers
        if answers.endswith("]"):
            answers += "\n```"
        if answers.startswith("```["):
            answers = answers.replace("```[", "```json\n[")

        # try:
        answers = re.findall(r"```json([\s\S]*)```", answers.split("</think>")[-1], re.S)[0]
        answers = eval(answers)
        if isinstance(answers, dict):
            answers = [answers]

        contents = []

        current_offset = 0  # 全局游标：记录当前解析到了 raw_content 的哪个位置

        for answer in answers:
            action_content = ""
            for key, value in answer.items():
                if "Action" in key and "Generate" in value:
                    action_content = answer["Generated Content"]
                    # TODO GenerateActionGenerate的内容不计算幻觉率
                    break

                elif "Action" in key and "Copy" in value:
                    try:
                        start = answer["Copy Start Character"].strip()
                        end = answer["Copy End Character"].strip()
                        length = answer["Copy Length"]

                        # 传入当前游标位置，并接收更新后的游标位置
                        match_res, new_offset = find_closest_substring(raw_content, start, end, int(length),
                                                                       current_offset)

                        if match_res:
                            action_content = match_res
                            current_offset = new_offset  # 成功匹配后，推进游标
                        else:
                            action_content = []
                    except Exception as e:
                        action_content = []
                    break

            if not action_content:
                return []

            if action_content:
                if isinstance(action_content, list):
                    action_content = "##".join(action_content)
                contents.append(action_content)
        if mode == "section":
            return contents
        return [''.join(contents)]

    async def batch_inference(self, datas, max_iters):
        """
        批量推理
        :param datas: 输入数据列表
        :return: 推理结果列表
        """
        total_outs = []
        batch_size = 10
        total_batch_datas = [datas[index:index + batch_size] for index in range(0, len(datas), batch_size)]
        for batch_datas in tqdm.tqdm(total_batch_datas, desc="Batch Inference"):
            # batch_datas = await self.section_inference(batch_datas, 0)
            batch_datas = self.for_mimic(batch_datas)
            batch_datas = await self.field_inference(batch_datas, 0)
            total_outs.extend(batch_datas)
        return total_outs

    def for_mimic(self, datas):
        new_datas = []
        for data in datas:
            content = data["content"]
            data["sub_contents"] = [{"sub_content": content, "sub_meta": data["meta"]}]
            new_datas.append(data)
        return new_datas

    async def section_inference(self, datas, max_iters):
        """
        对章节进行推理
        :param datas: 输入数据列表
        :return: 推理结果列表
        """

        PROMPT_agent = """You are a medical record section splitting agent responsible for splitting messy medical record documents into independent sections.

Given a medical record document, which may contain one or more medical record sections, please extract the "{}" section content from the following medical record document according to the medical record writing standards: {}.

Please pay attention to the following points during extraction:
(1) Complete the extraction of section content by using the "copy" and "generate" actions.
Among them, the "copy" action means you need to issue a "copy command", and simultaneously provide the "copy start character", "copy end character", and "copy length" parameters, thereby achieving the extraction of part of the section content.
The copy start character is the starting marker of the section, the copy end character is the ending marker of the section, and the copy start position and end position usually contain 5-8 characters. The copy length is the character length from the copy start position to the end position.
When it is necessary to skip certain content, you need to stop copying at the skipped position, then re-find the copy start position, and issue a "copy command" with the copy parameters.
When the field content to be copied is relatively short (less than 20 tokens), you need to use the "generate" action, meaning you need to issue a "generate command" and directly generate the target content, thereby achieving the extraction of the section content. Repeated copying or generating of the same content is not allowed.
During the section extraction process, if the section appears multiple times with repeated content, only the most complete version of the section needs to be extracted.
(2) Output the copy and generate processes in sequence according to the following format:
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generated Content": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
(3) When the section content to be extracted does not exist in the medical record text, return ```json[{{"Action1": "Generate", "Generated Content": "Not extracted relevant content"}}]```.

Medical Record Text:
{}"""
        PROMPT_agent_experience = """You are a medical record section splitting agent responsible for splitting messy medical record documents into independent sections.

Given a medical record document, which may contain one or more medical record sections, please extract the "{}" section content from the following medical record document according to the medical record writing standards: {}.

Please pay attention to the following points during extraction:
(1) Complete the extraction of section content by using the "copy" and "generate" actions.
Among them, the "copy" action means you need to issue a "copy command", and simultaneously provide the "copy start character", "copy end character", and "copy length" parameters, thereby achieving the extraction of part of the section content.
The copy start character is the starting marker of the section, the copy end character is the ending marker of the section, and the copy start position and end position usually contain 5-8 characters. The copy length is the character length from the copy start position to the end position.
When it is necessary to skip certain content, you need to stop copying at the skipped position, then re-find the copy start position, and issue a "copy command" with the copy parameters.
When the field content to be copied is relatively short (less than 20 tokens), you need to use the "generate" action, meaning you need to issue a "generate command" and directly generate the target content, thereby achieving the extraction of the section content. Repeated copying or generating of the same content is not allowed.
During the section extraction process, if the section appears multiple times with repeated content, only the most complete version of the section needs to be extracted.
(2) Output the copy and generate processes in sequence according to the following format:
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generated Content": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
(3) When the section content to be extracted does not exist in the medical record text, return ```json[{{"Action1": "Generate", "Generated Content": "Not extracted relevant content"}}]```.
(4) Based on existing validation, fully refer to the following experience to complete the task.

## Existing Experience:
{}

Medical Record Text:
{}"""
        PROMPT_agent_revise = """你是一个病历章节拆分智能体，负责将杂乱的病历文书拆分成独立的章节。
给你一份病历文书，其中可能包含一个或多个病历章节，请根据病历书写规范从下面的病历文书中抽取“{}”章节内容，{}。

抽取时请注意以下事项：
（1）通过使用“Copy”和“Generate”Action完成章节内容抽取。
其中，“Copy”Action指的是你需要发出“Copy命令”，并同时给出“Copy Start Character”、“Copy End Character”、“Copy Length”参数，从而实现部分章节内容的抽取。
其中Copy Start Character是章节的开始标志，Copy End Character是章节的结束标志，Copy起始位置和终止位置通常包含5-8个字符。Copy Length是从Copy起始位置到终止位置的字符长度。
当需要跳过某些内容时需要在跳过位置停止Copy，然后重新寻找Copy起始位置，并发出“Copy命令”，并给出Copy参数。
当需要Copy的字段内容较短时（小于20 tokens），需要使用“Generate”Action，即你需要发出“Generate命令”，并直接Generate目标内容，从而实现章节内容的抽取。不允许重复Copy或Generate相同内容。
抽取章节过程中，如果该章节出现多次且内容重复，仅需要抽取最完整的一份章节。
（2）按照下面的格式依次输出Copy和Generate过程：
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generate内容": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
（3）当需要抽取的章节内容在病历文本中不存在时，返回```json[{{"Action1": "Generate", "Generate内容": "Not extracted relevant content"}}]```。
（4）已经给出了“上一轮预测结果”和“上一轮错误情况”，请根据这些信息并避免发生相同的错误，确保本轮抽取结果正确无误。

## 上一轮识别时，下面的内容存在错误：
{}

具体错误是：
{}

病历文本：
{}"""
        # 初始化任务队列
        tasks = []
        for i, data in enumerate(datas):
            data["sub_contents"] = []  # 清空旧结果
            for section_name in section_explain:
                hos_name = data["meta"].get("hos_name")
                experiences = experience_add["sections"].get(hos_name, [])
                experiences_str = '\n'.join(experiences)
                tasks.append({
                    "data_idx": i,
                    "sec_name": section_name,
                    "meta": data["meta"],
                    "experience": experiences_str,
                    "error_hint": "",
                    "prev_content": ""
                })

        # 循环推理
        for attempt in range(max_iters + 1):
            if not tasks:
                break
            print(f"Section Extraction Attempt {attempt + 1}, Tasks count: {len(tasks)}")
            batch_query = []
            batch_raw_content = []

            # 1. 构建 Batch Query
            for task in tasks:
                data = datas[task["data_idx"]]
                raw_content = data["content"]
                # 预处理
                clean_raw_content = raw_content.replace("<sep1>", ":").replace("<sep2>", " ")
                if task["error_hint"]:
                    # 使用修正 Prompt
                    query = PROMPT_agent_revise.format(
                        task["sec_name"],
                        section_explain[task["sec_name"]],
                        task["prev_content"],
                        task["error_hint"],
                        clean_raw_content
                    )
                else:
                    # 使用初始 Prompt
                    if task["experience"]:
                        PROMPT = PROMPT_agent_experience if "extract" in self.ctype else PROMPT_agent_experience
                        query = PROMPT.format(
                            task["sec_name"],
                            section_explain[task["sec_name"]],
                            task["experience"],
                            clean_raw_content
                        )
                    else:
                        PROMPT = PROMPT_agent if "extract" in self.ctype else PROMPT_agent
                        query = PROMPT.format(
                            task["sec_name"],
                            section_explain[task["sec_name"]],
                            clean_raw_content
                        )

                batch_query.append(query)
                batch_raw_content.append(clean_raw_content)

            # 2. 执行推理
            if batch_query:
                async with AsyncOpenAI(base_url=self.url, api_key="EMPTY") as client:
                    batch_predicts = await process_queries_safe(batch_query, self.model_name, client)

                # 3. 解析结果并暂存
                for task, predict, raw_content in zip(tasks, batch_predicts, batch_raw_content):
                    if self.ctype == "extract":
                        part_sections = self._function_field_extract(predict)
                    elif self.ctype == "agent":
                        part_sections = self._function_agent_field(raw_content, predict, "section")
                    else:
                        part_sections = []
                    print(raw_content)
                    print(predict)
                    print(part_sections)
                    part_sections = [ps.replace("Not extracted relevant content", "") for ps in part_sections if
                                     ps.replace("Not extracted relevant content", "")]
                    # 同一个类型的章节可以识别出多个结果。
                    final_content = []
                    for sub_content in part_sections:
                        if sub_content:
                            final_content.append(sub_content)
                            task["meta"]["sec_name"] = task["sec_name"]
                            datas[task["data_idx"]]["sub_contents"].append({
                                "sub_content": sub_content,
                                "sub_meta": task["meta"]
                            })

                    task["sub_contents"] = final_content

        return datas

    async def field_inference(self, datas, max_retries=2):
        """
        对字段进行推理，包含错误检查和自我修正循环
        :param datas: 输入数据列表
        :param max_retries: 最大重试次数
        :return: 推理结果列表
        """

        PROMPT_agent = """You are a medical record field extraction agent responsible for completing the following task.

Please extract the "{}" field content from the following "{}" medical record. "{}" refers to "{}".

Please pay attention to the following points during extraction:
(1) Complete the extraction of section content by using the "copy" and "generate" actions.
Among them, the "copy" action means you need to issue a "copy command", and simultaneously provide the "copy start character", "copy end character", and "copy length" parameters, thereby achieving the extraction of part of the section content.
The copy start character is the starting marker of the section, the copy end character is the ending marker of the section, and the copy start position and end position usually contain 5-8 characters. The copy length is the character length from the copy start position to the end position.
When it is necessary to skip certain content, you need to stop copying at the skipped position, then re-find the copy start position, and issue a "copy command" with the copy parameters.
When the field content to be copied is relatively short (less than 20 tokens), you need to use the "generate" action, meaning you need to issue a "generate command" and directly generate the target content, thereby achieving the extraction of the section content. Repeated copying or generating of the same content is not allowed.
During the section extraction process, if the section appears multiple times with repeated content, only the most complete version of the section needs to be extracted.
(2) Output the copy and generate processes in sequence according to the following format:
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generated Content": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
(3) When the section content to be extracted does not exist in the medical record text, return ```json[{{"Action1": "Generate", "Generated Content": "Not extracted relevant content"}}]```.

Medical Record Text:
{}"""

        PROMPT_agent_experience = """You are a medical record field extraction agent responsible for completing the following task.

Please extract the "{}" field content from the following "{}" medical record. "{}" refers to "{}".

Please pay attention to the following points during extraction:
(1) Complete the extraction of section content by using the "copy" and "generate" actions.
Among them, the "copy" action means you need to issue a "copy command", and simultaneously provide the "copy start character", "copy end character", and "copy length" parameters, thereby achieving the extraction of part of the section content.
The copy start character is the starting marker of the section, the copy end character is the ending marker of the section, and the copy start position and end position usually contain 5-8 characters. The copy length is the character length from the copy start position to the end position.
When it is necessary to skip certain content, you need to stop copying at the skipped position, then re-find the copy start position, and issue a "copy command" with the copy parameters.
When the field content to be copied is relatively short (less than 20 tokens), you need to use the "generate" action, meaning you need to issue a "generate command" and directly generate the target content, thereby achieving the extraction of the section content. Repeated copying or generating of the same content is not allowed.
During the section extraction process, if the section appears multiple times with repeated content, only the most complete version of the section needs to be extracted.
(2) Output the copy and generate processes in sequence according to the following format:
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generated Content": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
(3) When the section content to be extracted does not exist in the medical record text, return ```json[{{"Action1": "Generate", "Generated Content": "Not extracted relevant content"}}]```.
(4) Based on existing validation, fully refer to the following experience to complete the task.

## Existing Experience:
{}

Medical Record Text:
{}"""

        PROMPT_agent_revise = """你是一个病历字段抽取智能体，请完成下面的任务。
请从下面的“{}”病历中抽取“{}”字段的内容， “{}”是“{}”。

抽取时请注意以下事项：
（1）通过使用“Copy”和“Generate”Action完成章节内容抽取。
其中，“Copy”Action指的是你需要发出“Copy命令”，并同时给出“Copy Start Character”、“Copy End Character”、“Copy Length”参数，从而实现部分章节内容的抽取。
其中Copy Start Character是章节的开始标志，Copy End Character是章节的结束标志，Copy起始位置和终止位置通常包含5-8个字符。Copy Length是从Copy起始位置到终止位置的字符长度。
当需要跳过某些内容时需要在跳过位置停止Copy，然后重新寻找Copy起始位置，并发出“Copy命令”，并给出Copy参数。
当需要Copy的字段内容较短时（小于20 tokens），需要使用“Generate”Action，即你需要发出“Generate命令”，并直接Generate目标内容，从而实现章节内容的抽取。不允许重复Copy或Generate相同内容。
抽取章节过程中，如果该章节出现多次且内容重复，仅需要抽取最完整的一份章节。
（2）按照下面的格式依次输出Copy和Generate过程：
```json[{{"Action1": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}},
{{"Action2": "Generate", "Generate内容": "_"}},
{{"Action3": "Copy", "Copy Start Character": "_", "Copy End Character": "_", "Copy Length": "_"}}, ...]```
（3）当需要抽取的章节内容在病历文本中不存在时，返回```json[{{"Action1": "Generate", "Generate内容": "Not extracted relevant content"}}]```。
（4）已经给出了“上一轮预测结果”和“上一轮错误情况”，请根据这些信息并避免发生相同的错误，确保本轮抽取结果正确无误。

## 上一轮预测结果：
{}

## 上一轮错误情况：
{}

病历文本：
{}
"""
        tasks = []
        for i, data in enumerate(datas):
            sub_contents = data["sub_contents"]
            for j, sub_content_dic in enumerate(sub_contents):
                # 确保fields字典存在
                if "fields" not in data["sub_contents"][j]:
                    data["sub_contents"][j]["fields"] = {}

                # sec_name = sub_content_dic["sub_meta"]["sec_name"]
                sec_name = "Discharge Summary"
                if sec_name in field_explain:
                    for field_name in field_explain[sec_name]:
                        experiences = experience_add["fields"].get(data["meta"].get("hos_name"), [])
                        experiences_str = '\n'.join(experiences)
                        tasks.append({
                            "data_idx": i,
                            "sub_content_idx": j,
                            "field_name": field_name,
                            "sec_name": sec_name,
                            "experience": experiences_str,
                            "error_hint": ""  # 初始无错误提示
                        })

        # 开始推理循环（初始 + 重试）
        for attempt in range(max_retries + 1):
            if not tasks:
                break
            print(f"Field Extraction Attempt {attempt + 1}, Tasks count: {len(tasks)}")

            # --- 1. 批量推理当前队列任务 ---
            batch_query = []
            batch_raw_content = []

            for task in tasks:
                data = datas[task["data_idx"]]
                sub_content_dic = data["sub_contents"][task["sub_content_idx"]]
                raw_content = sub_content_dic["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")

                # 构建带有错误提示的Prompt
                if task["error_hint"]:
                    query = PROMPT_agent_revise.format(
                        task["field_name"],
                        task["sec_name"],
                        task["field_name"],
                        field_explain[task["sec_name"]][task["field_name"]],
                        task["prev_field"],
                        task["error_hint"],
                        raw_content
                    )

                else:
                    if task["experience"]:
                        query = PROMPT_agent_experience.format(
                            task["field_name"],
                            task["sec_name"],
                            task["field_name"],
                            field_explain[task["sec_name"]][task["field_name"]],
                            task["experience"],
                            raw_content
                        )

                    else:
                        query = PROMPT_agent.format(
                            task["field_name"],
                            task["sec_name"],
                            task["field_name"],
                            field_explain[task["sec_name"]][task["field_name"]],
                            raw_content
                        )

                batch_query.append(query)
                batch_raw_content.append(raw_content)

            # 只有有任务才执行API调用
            if batch_query:
                print(batch_query[0])
                async with AsyncOpenAI(base_url=self.url, api_key="EMPTY") as client:
                    batch_predicts = await process_queries_safe(batch_query, self.model_name, client)

                # --- 2. 解析结果并写入datas ---
                for task, predict, raw_content in zip(tasks, batch_predicts, batch_raw_content):
                    if self.ctype == "extract":
                        field_content_list = self._function_field_extract(predict)
                    elif self.ctype == "agent":
                        field_content_list = self._function_agent_field(raw_content, predict, "field")
                    else:
                        continue

                    final_content = "##".join(field_content_list).replace("Not extracted relevant content", "")

                    # 更新数据中最 master 的存储位置
                    datas[task["data_idx"]]["sub_contents"][task["sub_content_idx"]]["fields"][
                        task["field_name"]] = final_content

            # 如果是最后一次尝试，不需要再做检查，直接退出
            if attempt == max_retries:
                break
            ## 保存当前轮次结果

            # --- 3. 错误检查 (Section Level) ---
            # 我们需要按 (data_idx, sub_content_idx) 聚合任务来检查整个章节
            # 找出本轮涉及到的所有章节索引
            affected_sections = set((t["data_idx"], t["sub_content_idx"]) for t in tasks)

            tasks = []  # 清空任务列表，准备装载下一轮需要重试的任务

            for d_idx, s_idx in affected_sections:
                current_sub = datas[d_idx]["sub_contents"][s_idx]
                raw_c = current_sub["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")
                current_fields = current_sub["fields"]
                sec_name = current_sub["sub_meta"]["sec_name"]

                # 调用外部提供的检查函数
                # 注意：error_check 需要根据实际导入路径调用
                try:
                    field_errors = await error_check(sec_name, raw_c, current_fields, self.url1, self.model_name1)
                except Exception as e:
                    print(f"Error check failed: {e}")
                    field_errors = {}

                # 哪个字段需要修订？错误类型是什么？{key：[error_types]}
                for field_name_prev_field, errors in field_errors.items():
                    field_name, prev_field = field_name_prev_field.split("##@@")
                    tasks.append({
                        "data_idx": d_idx,
                        "sub_content_idx": s_idx,
                        "field_name": field_name,
                        "sec_name": sec_name,
                        "error_hint": errors,
                        "prev_field": prev_field
                    })
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


async def check_section_quality(sec_name, raw_content, extract_sections, url, model_name):
    def parser_predict(predict, section_name):
        # 定位 markdown代码块中的json内容
        predict = json.loads(predict)["choices"][0]["message"]["content"]
        try:
            json_content = re.findall("\{[\s\S]*?\}", predict, re.S)
            # 尝试解析json数据
            data = json.loads(json_content[-1])
            if "是否杂糅" in data:
                predict_d = data["是否杂糅"]
                out = "在上一轮提取的章节“{}”中，其语义存在“{}”问题,也就是说当前提取的章节内容中可能包含两个或两个以上的章节。".format(
                    section_name, predict_d)

            elif "是否重复" in data:
                predict_d = data["是否重复"]
                out = "在上一轮提取的章节“{}”中，其语义存在“{}”问题,也就是说当前提取的所有章节内容，在章节层面存在重复。".format(
                    section_name, predict_d)

            elif "是否正确" in data:
                predict_d = data["是否正确"]
                out = "在上一轮提取的章节“{}”中，其语义存在“{}”问题,也就是说当前提取的内容并不属于该章节类型。".format(
                    section_name, predict_d)

            elif "语义是否完整" in data:
                predict_d = data["语义是否完整"]
                out = "在上一轮提取的章节“{}”中，其语义存在“{}”问题,也就是说原始病历文本中还有章节术语该章节类型。".format(
                    section_name, predict_d)

            else:
                return "错误"
            return out
        except:
            return "错误"

    queries = []
    for section_content in extract_sections:
        # 语义杂糅检查
        query1 = PROMPT_Conflation_Note.format(
            section_content,
            json.dumps(section_explain, ensure_ascii=False)
        )
        queries.append((sec_name, section_content, query1))
        # 语义重复检查
        query2 = PROMPT_Duplication_Note.format(
            section_content,
        )
        queries.append((sec_name, section_content, query2))
        # 名称错误检查
        query3 = PROMPT_Name_Error_Note.format(
            section_content,
            sec_name,
            json.dumps(section_explain, ensure_ascii=False)
        )
        queries.append((sec_name, section_content, query3))
        # 语义完整性检查
        query4 = PROMPT_incomplate_Note.format(
            raw_content,
            extract_sections,
            sec_name,
            section_explain[sec_name]
        )
        queries.append((sec_name, section_content, query4))
    async with AsyncOpenAI(base_url=url, api_key="EMPTY") as client:
        predicts = await process_queries_safe([item[-1] for item in queries], model_name, client)

    outs = collections.defaultdict(set)
    for (sec_name, sec_content, query), predict in zip(queries, predicts):
        error_type = parser_predict(predict, sec_name)

        if re.findall("不完整”|不正确”|(?<!不)杂糅”|(?<!不)重复”", error_type):
            # 由于section可能非常长，将其改成开始结尾的形式。
            # sec_content = sec_content[:20] + "......" + sec_content[-20:]
            outs[sec_name + "##@@" + sec_content].add(error_type)
    return outs


async def error_check(section_name, raw_content, fields, url, model_name):
    """
    返回：field_name,field_content,error_types
    :param raw_content:
    :param fields:
    :return:
    """

    def parser_predict(predict, field_name):
        # 定位 markdown代码块中的json内容
        predict = json.loads(predict)["choices"][0]["message"]["content"]
        try:
            json_content = re.findall("\{[\s\S]*?\}", predict, re.S)
            # 尝试解析json数据
            data = json.loads(json_content[-1])
            if "是否杂糅" in data:
                predict_d = data["是否杂糅"]
            elif "是否重复" in data:
                predict_d = data["是否重复"]
            elif "是否正确" in data:
                predict_d = data["是否正确"]
            elif "语义是否完整" in data:
                predict_d = data["语义是否完整"]
            else:
                return "错误"

            out = "字段“{}”的语义“{}”".format(field_name, predict_d)

            return out
        except:
            # 出现解析错误，同样稳健返回默认“否”
            return "错误"

    queries = []
    for field_name, field_content in fields.items():
        if field_content:

            # 语义杂糅检查
            query1 = PROMPT_Conflation_Section.format(
                section_name,
                raw_content,
                field_name,
                field_content,
                json.dumps(field_explain[section_name],
                           ensure_ascii=False)
            )
            queries.append((field_name, field_content, query1))
            # 语义重复检查
            query2 = PROMPT_Duplication_Section.format(
                field_name,
                field_content,
                json.dumps(fields, ensure_ascii=False)
            )
            queries.append((field_name, field_content, query2))
            # 名称错误检查
            query3 = PROMPT_Name_Error_Section.format(
                field_content,
                field_name, field_explain[section_name][field_name],
                json.dumps(field_explain[section_name], ensure_ascii=False)
            )
            queries.append((field_name, field_content, query3))

            # 语义完整性检查
            query4 = PROMPT_incomplate_Section.format(
                raw_content,
                field_name, field_explain[section_name][field_name],
                field_content
            )
            queries.append((field_name, field_content, query4))
        else:
            # 语义完整性检查
            query4 = PROMPT_incomplate_Section.format(
                raw_content,
                field_name, field_explain[section_name][field_name],
                field_content
            )
            queries.append((field_name, field_content, query4))

    async with AsyncOpenAI(base_url=url, api_key="EMPTY") as client:
        predicts = await process_queries_safe([item[-1] for item in queries], model_name, client)

    outs = collections.defaultdict(set)
    for (field_name, field_content, query), predict in zip(queries, predicts):
        error_type = parser_predict(predict, field_name)
        if re.findall("不完整|不正确|(?<!不)杂糅|(?<!不)重复", error_type):
            if field_content:
                outs[field_name + "##@@" + field_content].add(error_type)

    return outs


if __name__ == "__main__":
    # 示例用法,这个推理脚本是不做HI结构推理的。直接做CopyGenerate或直接Generate
    import asyncio

    # model_name = "qwen-agent-14b-grpo-experience"
    # url = "http://10.0.0.228:8001/v1"

    model_name = "qwen-agent-14b"
    # model_name = "qwen-agent-7b"
    # model_name = "qwen-extract-14b"
    url = "http://10.0.0.228:8000/v1"

    model_name1 = "qwen32b-train"
    url1 = "http://10.0.0.228:8822/v1"
    max_iters = 3

    data_ctype = "mimic"  # test, jiwai, mimic
    if data_ctype == "test":
        datas = read_datas("train_test_datas/base/v1/test_data.json")
        # datas = read_datas("train_test_datas/base/v1/debug.json")
    elif data_ctype == "jiwai":
        datas = read_datas("train_test_datas/base/v1/jiwai_data.json")
    else:
        datas = read_datas("datas/note_sample_50_已标注_加噪.json")

    agent = PiplineInference(model_name, url, model_name1, url1, "agent")
    extract = PiplineInference(model_name, url, model_name1, url1, "extract")
    loop = asyncio.new_event_loop()  # 创建新的 loop 避免冲突
    asyncio.set_event_loop(loop)
    if "agent" in model_name:
        os.makedirs("./predicts/{}/".format(model_name), exist_ok=True)
        agent_results = loop.run_until_complete(agent.batch_inference(datas, max_iters))
        write_results_to_file(agent_results, "./predicts/{}/{}_results_wo_base.json".format(model_name, data_ctype, max_iters))
    else:
        os.makedirs("./predicts/{}/".format(model_name), exist_ok=True)
        extract_results = loop.run_until_complete(extract.batch_inference(datas, max_iters))
        write_results_to_file(extract_results, "./predicts/{}/{}_results_wo_base.json".format(model_name, data_ctype, max_iters))
