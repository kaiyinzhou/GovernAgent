# -*- coding:utf-8 -*-
"""

启动服务:
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-72B-Instruct
--model ./models/Qwen2-72B-Instruct --tensor-parallel-size 2 --max_model_len 8192 --gpu_memory_utilization 0.95
"""
import asyncio
import json
import os

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import re
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer

tokenizer_kwargs = {
    "cache_dir": None,
    "use_fast": False,
    "revision": "main",
    "use_auth_token": None,
}

# tokenizer = AutoTokenizer.from_pretrained("../tokenizer", **tokenizer_kwargs)

tokenizer = AutoTokenizer.from_pretrained("/home/share/sft/qwen2.5/qwen2.5-14b")
#
# tokenizer = AutoTokenizer.from_pretrained("/home/zhouky/Llama3/8b")


async def async_query_openai(query, url, model_name):
    # if model_name in {"Qwen2.5-summary"}:
    # qwen2.5-14b
    JinJa = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    # llama
    # JinJa = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    max_length = 8192
    openai_api_key = "EMPTY"
    openai_api_base = url

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        # timeout=httpx.Timeout(connect=5.0, read=5000.0, write=300.0, pool=5.0)
    )
    messages = [
        {"role": "user", "content": query},
    ]
    query_id = tokenizer.apply_chat_template(messages, tokenize=True, chat_template=JinJa, add_generation_prompt=True)
    try:

        chat_response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_length - len(query_id),
        )
        return chat_response.model_dump_json()
    except:
        return None


async def async_process_queries(queries, url, model_name):
    results = await asyncio.gather(*(async_query_openai(query, url, model_name) for query in queries))
    return results


from collections import Counter
import jieba  # 用于中文分词


def tokenize_chinese(text, mode="word"):
    """
    Tokenize Chinese text based on the mode.
    - mode='word': Tokenize into words (using Jieba)
    - mode='char': Treat each character as a token
    """
    if mode == "word":
        return list(jieba.cut(text))
    elif mode == "char":
        return list(text)
    else:
        raise ValueError("Mode must be 'word' or 'char'")


def n_gram_counter(text, n):
    """Create n-grams from a tokenized sequence."""
    return [tuple(text[i:i + n]) for i in range(len(text) - n + 1)]


def rouge_n(reference, generated, n):
    """Calculate ROUGE-N score."""
    ref_ngrams = Counter(n_gram_counter(reference, n))
    gen_ngrams = Counter(n_gram_counter(generated, n))

    # Count overlapping n-grams
    matches = sum((ref_ngrams & gen_ngrams).values())
    total_reference = sum(ref_ngrams.values())

    # Avoid division by zero
    if total_reference == 0:
        return 0.0

    return matches / total_reference


def lcs(reference, generated):
    """Calculate the Longest Common Subsequence (LCS) length."""
    ref_len = len(reference)
    gen_len = len(generated)

    # Create a table for dynamic programming
    dp_table = [[0] * (gen_len + 1) for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        for j in range(1, gen_len + 1):
            if reference[i - 1] == generated[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1
            else:
                dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

    return dp_table[ref_len][gen_len]


def rouge_l(reference, generated, beta=1.0):
    """Calculate ROUGE-L score based on LCS."""
    lcs_len = lcs(reference, generated)
    ref_len = len(reference)
    gen_len = len(generated)

    if ref_len == 0 or gen_len == 0:
        return 0.0, 0.0, 0.0

    recall = lcs_len / ref_len
    precision = lcs_len / gen_len

    if recall + precision == 0:
        return 0.0, 0.0, 0.0

    f1_score = ((1 + beta ** 2) * recall * precision) / (beta ** 2 * precision + recall)
    return recall, precision, f1_score


def weighted_lcs(reference, generated, weight_factor=1.0):
    """Calculate Weighted Longest Common Subsequence (Weighted LCS)."""
    ref_len = len(reference)
    gen_len = len(generated)

    # Create a table for dynamic programming with weights
    dp_table = [[0] * (gen_len + 1) for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        for j in range(1, gen_len + 1):
            if reference[i - 1] == generated[j - 1]:
                # Continuous matching gets higher weights
                dp_table[i][j] = dp_table[i - 1][j - 1] + (weight_factor if i > 1 and j > 1 else 1)
            else:
                dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

    return dp_table[ref_len][gen_len]


def rouge_w(reference, generated, alpha=1.0):
    """Calculate ROUGE-W score."""
    weighted_lcs_len = weighted_lcs(reference, generated, alpha)
    total_possible = len(reference) + len(generated)

    if total_possible == 0:
        return 0.0

    return (2 * weighted_lcs_len) / total_possible


def find_closest_substring(string, start_str, end_str, length):
    string = string.replace(":", "：")
    start_str = start_str.replace(":", "：")
    end_str = end_str.replace(":", "：")
    # 构造正则表达式，匹配以start_str 开头，end_str结尾的子串
    # pattern = re.escape(start_str) + "([\s\S]*)" + re.escape(end_str)
    # 去除所有空格。

    starts = find_all_start_indices(start_str, string)
    ends = find_all_end_indices(end_str, string)
    matches = []
    for start in starts:
        for end in ends:
            if start < end:
                match = string[start:end + 1]
                matches.append(match)

    if not matches:
        return None  # 如果没有匹配的子串，返回 None

    # 构造所有满足条件的准确子串列表
    substrings = [match for match in matches]
    # 找出与给定长度最接近的子串，并记录其偏差
    closest_substring = None
    closest_diff = float('inf')

    for sub in substrings:
        sub_tokens = tokenizer(sub)["input_ids"]
        diff = abs(len(sub_tokens) - length)
        if diff < closest_diff:
            closest_substring = sub
            closest_diff = diff
    return closest_substring

def field_process(raw_content, answers):

    answers = json.loads(answers[0])["choices"][0]["message"]["content"]
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
                contents.append("".join(action_content))
            contents.append(action_content)

    return [''.join(contents)]


def find_all_start_indices(a, C):
    indices = []
    start = 0
    while start < len(C):
        start = C.find(a, start)  # 查找 `a` 在 `C` 中从索引 `start` 开始的下一个位置
        if start == -1:  # 如果找不到，就退出循环
            break
        indices.append(start)  # 添加找到的索引
        start += 1  # 更新起始位置，避免重复查找
    return indices


def find_all_end_indices(b, C):
    indices = []
    start = 0
    while start < len(C):
        start = C.find(b, start)  # 查找 `b` 在 `C` 中从索引 `start` 开始的下一个位置
        if start == -1:  # 如果找不到，就退出循环
            break
        end_index = start + len(b) - 1  # 计算结束位置索引
        indices.append(end_index)  # 添加结束索引
        start += 1  # 更新起始位置，避免重复查找
    return indices

