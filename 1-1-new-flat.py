# -*- coding:utf-8 -*-

import collections
import re

import tqdm

from utils import *

section_explain = json.load(open("./configs/label_explain.json", encoding='utf-8'))["Section"]
field_explain = json.load(open("./configs/label_explain.json", encoding='utf-8'))["Field"]


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

    async def batch_inference(self, datas, max_iters):
        """
        批量推理
        :param datas: 输入数据列表
        :return: 推理结果列表
        """
        total_outs = []
        batch_size = 5
        total_batch_datas = [datas[index:index + batch_size] for index in range(0, len(datas), batch_size)]
        for batch_datas in tqdm.tqdm(total_batch_datas, desc="Batch Inference"):
            batch_datas = await self.section_inference(batch_datas)
            batch_datas = await self.field_inference(batch_datas, max_iters)
            total_outs.extend(batch_datas)
        return total_outs

    async def section_inference(self, datas):
        """
        直接用章节结果替代推理结果，说明这个阶段实际没有推理
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
            if action_content:
                if isinstance(action_content, list):
                    action_content = "##".join(action_content)
                contents.append(action_content)
        return [''.join(contents)]

    async def field_inference(self, datas, max_retries=2):
        """
        对字段进行推理，包含错误检查和自我修正循环
        :param datas: 输入数据列表
        :param max_retries: 最大重试次数
        :return: 推理结果列表
        """
#         PROMPT_extract = """你是一个病历字段抽取智能体，请完成下面的任务。
# 请从下面的“{}”病历中抽取“{}”字段的内容，“{}”是“{}”。
# 抽取时请注意以下事项：
# （1）当字段内容重复出现时，只需抽取一次。
# （2）对于病历中的乱码部分，不需要抽取。
# （3）当需要抽取的内容并非连续出现时，请确保抽取内容完整，不连续的内容用“##”号连接。保留关键词，例如“性别”、“年龄”等。
# （4）按照下面的格式进行抽取：```json[{{"字段名称": "_", "字段内容": "_"}}]```。
# （5）当需要抽取的字段在病历文本中不存在时，“字段内容”部分填写“未抽取到相关内容”。
#
# 病历文本：
# {}"""
        PROMPT_extract = """You are a medical record field extraction agent. Please complete the following task.

Extract the content of the "{}" field from the medical record within "{}". "{}" refers to "{}".
Please pay attention to the following points during extraction:

(1) If the field content appears repeatedly, extract it only once.
(2) Do not extract garbled parts in the medical record.
(3) If the content to be extracted is not continuous, ensure the extracted content is complete, and connect discontinuous parts with "##". Retain keywords such as "gender" and "age".
(4) Extract in the following format: ```json[{{"field_name": "_", "field_content": "_"}}]```.
(5) If the required field does not exist in the medical record text, fill in "未抽取到相关内容" (content not found) in the "field_content" section.

Medical record text:
{}
"""
        PROMPT_extract_revise = """你是一个病历字段抽取智能体，请完成下面的任务。
请从下面的“{}”病历中抽取“{}”字段的内容，“{}”是“{}”。
抽取时请注意以下事项：
（1）当字段内容重复出现时，只需抽取一次。
（2）对于病历中的乱码部分，不需要抽取。
（3）当需要抽取的内容并非连续出现时，请确保抽取内容完整，不连续的内容用“##”号连接。保留关键词，例如“性别”、“年龄”等。
（4）按照下面的格式进行抽取：```json[{{"字段名称": "_", "字段内容": "_"}}]```。
（5）当需要抽取的字段在病历文本中不存在时，“字段内容”部分填写“未抽取到相关内容”。
（6） 已经给出了“上一轮预测结果”和“上一轮错误情况”，请根据这些信息提取结果并避免发生相同的错误，确保本轮抽取结果正确无误。
## 上一轮预测结果：
{}

## 上一轮错误情况：
{}

病历文本：
{}"""

        # 初始化：构建初始任务队列
        # 结构: {"data_idx": int, "sub_content_idx": int, "field_name": str, "error_history": str}
        tasks = []
        for i, data in enumerate(datas):
            sub_contents = data["sub_contents"]
            for j, sub_content_dic in enumerate(sub_contents):
                # 确保fields字典存在
                if "fields" not in data["sub_contents"][j]:
                    data["sub_contents"][j]["fields"] = {}

                sec_name = sub_content_dic["sub_meta"]["sec_name"]
                if sec_name in field_explain:
                    for field_name in field_explain[sec_name]:
                        tasks.append({
                            "data_idx": i,
                            "sub_content_idx": j,
                            "field_name": field_name,
                            "sec_name": sec_name,
                            "error_hint": ""  # 初始无错误提示
                        })

        # 开始推理循环（初始 + 重试）
        for attempt in range(max_retries + 1):
            if not tasks:
                break

            print(f"Extraction Attempt {attempt + 1}, Tasks count: {len(tasks)}")

            # --- 1. 批量推理当前队列任务 ---
            batch_query = []
            batch_raw_content = []

            for task in tasks:
                data = datas[task["data_idx"]]
                sub_content_dic = data["sub_contents"][task["sub_content_idx"]]
                raw_content = sub_content_dic["sub_content"].replace("<sep1>", ":").replace("<sep2>", " ")

                # 构建带有错误提示的Prompt
                if task["error_hint"]:
                    query = PROMPT_extract_revise.format(
                        task["sec_name"],
                        task["field_name"],
                        task["field_name"],
                        field_explain[task["sec_name"]][task["field_name"]],
                        task["prev_field"],
                        task["error_hint"],
                        raw_content
                    )

                else:
                    query = PROMPT_extract.format(
                        task["sec_name"],
                        task["field_name"],
                        task["field_name"],
                        field_explain[task["sec_name"]][task["field_name"]],
                        raw_content
                    )

                batch_query.append(query)
                batch_raw_content.append(raw_content)

            # 只有有任务才执行API调用
            if batch_query:
                print("######")
                print(batch_query[0])
                batch_predicts = await async_process_queries(batch_query, self.url, self.model_name)
                print(batch_predicts[0])


                # --- 2. 解析结果并写入datas ---
                for task, predict, raw_content in zip(tasks, batch_predicts, batch_raw_content):
                    if self.ctype == "extract":
                        field_content_list = self._function_field_extract(predict)
                    elif self.ctype == "agent":
                        field_content_list = self._function_agent_field(raw_content, predict)
                    else:
                        continue

                    final_content = "##".join(field_content_list).replace("未抽取到相关内容", "")

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


from prompt_temp import *


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
                json.dumps(fields, ensure_ascii=False)
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
    predicts = await async_process_queries([item[-1] for item in queries], url, model_name)

    outs = collections.defaultdict(set)
    for (field_name, field_content, query), predict in zip(queries, predicts):
        error_type = parser_predict(predict, field_name)
        if re.findall("不完整|不正确|(?<!不)杂糅|(?<!不)重复", error_type):
            if "重复" not in error_type:
                outs[field_name + "##@@" + field_content].add(error_type)
    return outs


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

    # model_name = "qwen-extract-7b"
    model_name = "qwen-extract-7b-sft"
    url = "http://10.0.0.228:8000/v1"

    model_name1 = "qwen70b"
    url1 = "http://10.0.0.228:8822/v1"
    # model_name1 = "qwen-extract-14b"
    # url1 = "http://10.0.0.228:8000/v1"
    max_iters = 0

    data_ctype = "mimic"
    if data_ctype == "test":
        datas = read_datas("train_test_datas/base/v1/test_data.json")
        # datas = read_datas("train_test_datas/base/v1/debug.json")
    elif data_ctype == "jiwai":
        datas = read_datas("train_test_datas/base/v1/jiwai_data.json")
    else:
        datas = read_datas("datas/note_sample_50_已标注_加噪.json")

    agent = PiplineInference(model_name, url, model_name1, url1, "agent")
    extract = PiplineInference(model_name, url, model_name1, url1, "extract")
    #
    if "agent" in model_name:
        os.makedirs("./predicts/{}/".format(model_name), exist_ok=True)

        agent_results = asyncio.run(agent.batch_inference(datas, max_iters))
        write_results_to_file(agent_results,
                              "./predicts/{}/{}_results_{}.json".format(model_name, data_ctype, max_iters))
    else:
        os.makedirs("./predicts/{}/".format(model_name), exist_ok=True)
        extract_results = asyncio.run(extract.batch_inference(datas, max_iters))
        write_results_to_file(extract_results,
                              "./predicts/{}/{}_results_{}.json".format(model_name, data_ctype, max_iters))

